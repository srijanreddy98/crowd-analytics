const { getEmotion, getPeople} = require('./serivces/python');
const express = require('express');
const mongoose = require('mongoose');
const cookieSession = require('cookie-session');
const path = require('path');
const logger = require('morgan');
const cookieParser = require('cookie-parser');
var bodyParser = require('body-parser');
const { routes } = require('./Routes/authRoutes');
const cluster = require('cluster');

if (cluster.isMaster) {
    cluster.fork();
    // cluster.fork();
    // cluster.fork();
} else {
    // Running the python script
    getEmotion();
    getPeople();

    // Setting Mongoose Parameters and connecting to database
    mongoose.Promise = global.Promise;
    mongoose.connect('mongodb://localhost:27017/crowdAnalytics');

    const app = express();

    //Middlewares
    app.use(logger('dev'));
    app.use(bodyParser.json({ limit: '50mb' }));
    app.use(bodyParser.urlencoded({
        limit: '50mb',
        extended: true
    }));

    // Front End Route
    app.use(express.static(__dirname + '/dist2'));
    app.get('/dashboard', function (req, res) {
        res.sendFile(path.join(__dirname + '/dist2', 'index.html'));
    });

    // Initializing API Routes
    routes(app);

    // Starting Server
    const PORT = process.env.PORT || 3000;
    app.listen(PORT, () => console.log(`Server is up on port:${PORT}`));
}