const mongoose = require('mongoose');
const { Schema } = mongoose;

const ideaSchema = new Schema({
    timeStamp: {
        type: String,
        default: null
    },
    title: {
        type: String,
        default: null
    },
    description: String,
    language: {
        type: String,
        default: 'en'
    },
    status: {
        type: String,
        default: 'good'
    }
});
const intrestedSchema = new Schema({
    timeStamp: {
        type: String,
        default: Date.now()
    },
    countA: Number,
    countNA: Number
});

var Idea = mongoose.model('idea', ideaSchema);
var Intrested = mongoose.model('intrested', intrestedSchema)
module.exports = { Idea , Intrested};
