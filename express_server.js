const express = require('express');
const axios = require('axios');
const app = express();
const port = 3000;

app.use(express.json());

app.post('/predict', async (req, res) => {
    const features = req.body.features;  // Expecting features in the request body
    try {
        const response = await axios.post('http://localhost:5000/predict', { features });
        res.json(response.data);  // Forward the prediction from the Flask server
    } catch (error) {
        res.status(500).send(error.message);
    }
});

app.listen(port, () => {
    console.log(`Express server is running on port ${port}`);
});
