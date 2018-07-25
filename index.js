const {
    getAttentive, getEmotion, getPeople
} = require('./serivces/python');
// getAttentive();
getEmotion();
// getPeople();
const express = require('express');
const mongoose = require('mongoose');
const cookieSession = require('cookie-session');
const path = require('path');
const logger = require('morgan');
const cookieParser = require('cookie-parser');
var bodyParser = require('body-parser');
const { routes } = require('./Routes/authRoutes');
var fs = require('fs');
const { Idea } = require('./Models/models');
mongoose.Promise = global.Promise;
mongoose.connect('mongodb://srijanreddy98:opexai@ds119110.mlab.com:19110/crowdanalytics');
// mongoose.connect('mongodb://localhost:27017/crowdAnalytics');
// function to encode file data to base64 encoded string
function base64_encode(file) {
    // read binary data
    var bitmap = fs.readFileSync(file);
    // convert binary data to base64 encoded string
    return new Buffer(bitmap).toString('base64');
}

const app = express();

app.use(logger('dev'));
app.use(bodyParser.json({ limit: '50mb' }));
app.use(bodyParser.urlencoded({
     limit: '50mb',
    extended: true 
}));
app.use(express.static(__dirname + '/dist2'));
app.get('/dashboard', function (req, res) {
            res.sendFile(path.join(__dirname + '/dist2', 'index.html'));
        });
// app.get('/insurance', function (req, res) {
//     res.sendFile(path.join(__dirname + '/chatbots/Insurance', 'index.html'));
// });
// app.get('/netbanking', function (req, res) {
//     res.sendFile(path.join(__dirname + '/chatbots/Netbanking', 'index.html'));
// });
routes(app);
app.get('/api/img', (req, res) => {
    var base = base64_encode('fina.jpg');
    res.send({
        base
    });
});
const PORT = process.env.PORT || 3000;
app.listen(PORT,() => console.log(`Server is up on port:${PORT}`));