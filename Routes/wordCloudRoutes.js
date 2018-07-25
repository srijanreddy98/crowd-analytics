const fs = require('fs');
const json2xls = require('json2xls');
const { Idea } = require('../Models/models');
const { runShell } = require('../serivces/python');
const mongoose = require('mongoose');
var ObjectId = mongoose.Types.ObjectId();
var wordCloudRoutes = (app) => {
    app.get('/api/wordcloud', (req, res) => {
        var img = fs.readFileSync(__dirname + '/word1.png', {
            encoding: 'base64'
        });
        res.send(img);
    });
    app.get('/api/idea', (req, res) => {
        Idea.find({}).then(
            docs => res.send(docs),
            err => res.send(err)
        )
    });
    app.post('/api/idea', (req, res) => {
        console.log(req.body);
        // res.send(req.body);
        idea = new Idea({
            title: req.body.title,
            description: req.body.description
        });
        idea.save().then(
            re => {
                Idea.find({}).then(
                    (docs) => {
                        var arr = [];
                        for (i of docs) {
                            var k = {
                                SuggestionID: i['_id'],
                                Created: ObjectId.getTimestamp(i['_id']),
                                TITLE: i.title,
                                Description: i.description,
                                Language: 'en',
                                REFERENCE: "64",
                                Status: 'good',
                            };
                            arr.push(k)
                        }
                        var xls = json2xls(arr);
                        fs.writeFileSync(__dirname + '/../idea_export.xlsx', xls, 'binary');
                        runShell();
                    });
            });
        res.send(req.body);
    });
}

module.exports = {
    wordCloudRoutes
}