const { imageRoutes } = require('./imageRoutes');
const { wordCloudRoutes } = require('./wordCloudRoutes');
const { graphDataRoutes } = require('./graphDataRoutes');
const { realTimeFaceDataRoutes } = require('./realTimeFaceData');
var routes = (app) => {
    imageRoutes(app);
    wordCloudRoutes(app);
    graphDataRoutes(app);
    realTimeFaceDataRoutes(app);
}
module.exports = {
    routes
}
