async function handleEvent(event) {
    if (event.type !== 'message' || event.message.type !== 'image') {
        return Promise.resolve(null);
    }
    try {
        const imageBuffer = await getImageBufferFromLine(event.message.id);

        const imageTensor = tf.node.decodeImage(imageBuffer, 3)
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .div(tf.scalar(127.5))
            .sub(tf.scalar(1))
            .expandDims();

        const predictionResult = await model.predict(imageTensor).data();

        let bestPrediction = { className: 'ไม่รู้จัก', probability: 0 };
        for (let i = 0; i < predictionResult.length; i++) {
            if (predictionResult[i] > bestPrediction.probability) {
                bestPrediction.probability = predictionResult[i];
                bestPrediction.className = classNames[i];
            }
        }

        const confidence = Math.round(bestPrediction.probability * 100);

        // ======= Flex Message =======
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
                hero: {
                    type: 'image',
                    url: `https://example.com/your-preview-image.png`, // ใส่ภาพตัวอย่างหรือภาพผู้ใช้
                    size: 'full',
                    aspectRatio: '20:13',
                    aspectMode: 'cover'
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
                            type: 'box',
                            layout: 'baseline',
                            contents: [
                                {
                                    type: 'text',
                                    text: `ความแม่นยำ: ${confidence}%`,
                                    size: 'sm',
                                    color: '#666666',
                                    flex: 0
                                }
                            ]
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
