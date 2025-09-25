// index.js
const express = require('express');
const line = require('@line/bot-sdk');
const tf = require('@tensorflow/tfjs-node');

const config = {
    channelAccessToken: process.env.CHANNEL_ACCESS_TOKEN,
    channelSecret: process.env.CHANNEL_SECRET,
};

const modelUrl = 'https://teachablemachine.withgoogle.com/models/AFn6FL5Uf/model.json';
const classNames = ['with_mask', 'without_mask'];

const app = express();
const client = new line.Client(config);

let model;

// ==================== Webhook ====================
app.post('/webhook', line.middleware(config), (req, res) => {
    Promise.all(req.body.events.map(handleEvent))
        .then((result) => res.json(result))
        .catch((err) => {
            console.error(err);
            res.status(500).end();
        });
});

// ==================== Handle Event ====================
async function handleEvent(event) {
    if (event.type !== 'message' || event.message.type !== 'image') {
        return Promise.resolve(null);
    }

    try {
        const imageBuffer = await getImageBufferFromLine(event.message.id);

        // ==================== Preprocess Image ====================
        const imageTensor = tf.node.decodeImage(imageBuffer, 3)
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .div(tf.scalar(127.5))  // ปรับค่าสี
            .sub(tf.scalar(1))
            .expandDims();

        // ==================== Predict ====================
        const predictionResult = await model.predict(imageTensor).data();

        let bestPrediction = { className: 'ไม่รู้จัก', probability: 0 };
        for (let i = 0; i < predictionResult.length; i++) {
            if (predictionResult[i] > bestPrediction.probability) {
                bestPrediction.probability = predictionResult[i];
                bestPrediction.className = classNames[i];
            }
        }

        const confidence = Math.round(bestPrediction.probability * 100);

        // ==================== Flex Message ====================
        const flexMessage = {
            type: 'flex',
            altText: 'ผลการทำนาย',
            contents: {
                type: 'bubble',
                header: {
                    type: 'box',
                    layout: 'vertical',
                    contents: [
                        {
                            type: 'text',
                            text: 'ผลการทำนายหน้ากาก',
                            weight: 'bold',
                            color: '#ffffff',
                            size: 'lg'
                        }
                    ],
                    backgroundColor: '#4CAF50',
                    paddingAll: '10px'
                },
                body: {
                    type: 'box',
                    layout: 'vertical',
                    spacing: 'md',
                    contents: [
                        {
                            type: 'text',
                            text: `ฉันคิดว่ารูปนี้คือ: ${bestPrediction.className}`,
                            weight: 'bold',
                            size: 'md',
                            wrap: true
                        },
                        {
                            type: 'text',
                            text: `ความแม่นยำ: ${confidence}%`,
                            size: 'sm',
                            color: '#666666'
                        },
                        {
                            type: 'box',
                            layout: 'vertical',
                            margin: 'md',
                            contents: [
                                {
                                    type: 'filler',
                                    flex: confidence
                                }
                            ],
                            backgroundColor: '#D3D3D3',
                            height: '10px'
                        }
                    ]
                }
            }
        };

        return client.replyMessage(event.replyToken, flexMessage);

    } catch (error) {
        console.error(error);
        return client.replyMessage(event.replyToken, {
            type: 'text',
            text: 'ขออภัยค่ะ เกิดข้อผิดพลาดบางอย่าง'
        });
    }
}

// ==================== Get Image Buffer ====================
function getImageBufferFromLine(messageId) {
    return new Promise((resolve, reject) => {
        client.getMessageContent(messageId)
            .then((stream) => {
                const chunks = [];
                stream.on('data', (chunk) => { chunks.push(chunk); });
                stream.on('error', (err) => { reject(err); });
                stream.on('end', () => { resolve(Buffer.concat(chunks)); });
            });
    });
}

// ==================== Start Server ====================
async function startServer() {
    try {
        console.log('Loading model...');
        model = await tf.loadLayersModel(modelUrl);
        console.log('Model loaded!');

        const port = process.env.PORT || 3000;
        app.listen(port, () => {
            console.log(`Bot is ready on port ${port}`);
        });

    } catch (error) {
        console.error('Failed to load model:', error);
    }
}

startServer();
