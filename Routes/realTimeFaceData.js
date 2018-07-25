const fs = require('fs');
var realTimeFaceDataRoutes = (app) => {
    app.get('/api/faceData', (req, res) => {
        var data = JSON.parse(fs.readFileSync(__dirname + '/../4forces.json'));
        res.send(data);
    });
    app.get('/api/attentive', (req, res) => {
        var data = JSON.parse(fs.readFileSync(__dirname + '/../4forces2.json'));
        res.send(data);
    });
}

module.exports = {
    realTimeFaceDataRoutes
}