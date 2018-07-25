const fs = require('fs');
const {move} = require('../serivces/copy');
var XLSX = require('xlsx');
const json2xls = require('json2xls');
const { Idea, Intrested } = require('../Models/models');
const { runShell, getEmotion } = require('../serivces/python');
const mongoose = require('mongoose');
var ObjectId = mongoose.Types.ObjectId();
var user ;
var routes = (app) => {
    app.get('/api/wordcloud', (req, res) => {
        var img = fs.readFileSync(__dirname + '/word1.png', { encoding: 'base64' });
        // console.log(img);
        // var bitmap = fs.readFileSync(__dirname + '/../word1.png');
        // // convert binary data to base64 encoded string
        // res.send(new Buffer(bitmap).toString('base64'));
        res.send(img);
    });
    app.post('/api/images', (req, res) => {
        fs.writeFile(__dirname + '/../FINAL/src/' + 'photo.jpg', req.body.image, { encoding: 'base64' }, function (err) {
            if (err) console.log(err)
            console.log('File created');
            res.send({success: 'success'})
        });
        fs.writeFile(__dirname + '/../Attention/' + 'photo.jpg', req.body.image, { encoding: 'base64' }, function (err) {
            if (err) console.log(err);
        });
        fs.writeFile(__dirname + '/../people_count/' + 'photo.jpg', req.body.image, { encoding: 'base64' }, function (err) {
            if (err) console.log(err);
        });
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
                            Language:'en',
                            REFERENCE: "64",
                            Status: 'good',
                        }; arr.push(k)
                    }
                    var xls = json2xls(arr);
                    fs.writeFileSync(__dirname + '/../idea_export.xlsx', xls, 'binary');
                    runShell();
            });
    });
    res.send(req.body);
});
    app.get('/api/user', (req, res) => {
        console.log(req.query);
        user = req.query.name;
        res.send(req.query);
    });
    app.get('/api/currentuser', (req, res) => {
        res.send(user);
    });
    app.post('/api/statistic', (req, res) => {
        console.log(req.body);
        // var ins = Intrested({
        //     countA: 0,
        //     countNA: 0
        // });
        // ins.save();
        Intrested.find().sort({ "_id": -1 }).limit(1).then((post) => {
            if (post[0]) {
                const now = Date.now();
                var t = ((now - +post[0].timeStamp) / 600000);
                console.log(t);
                if (t < 1) {
                    countA = req.body.countA;
                    countNA = req.body.countNA;
                    console.log('here');
                    Intrested.findByIdAndUpdate(post[0]['_id'], { $set: { timeStamp: post[0].timeStamp, countA: countA, countNA: countNA } }, { new: true }).then(
                        docs => res.send(docs),
                        err => res.send(err)
                    );
                } else {
                    var time = +post[0].timeStamp;
                    while (t > 1) {
                        console.log('here2');
                        var i = Intrested({
                            timeStamp: time + 600000,
                            countA: 0,
                            countNA: 0
                        });
                        i.save();
                        time = time + 600000;
                        t = t - 1;
                    }
                    countA = req.body.countA;
                    countNA = req.body.countNA;
                    Intrested.findByIdAndUpdate(post[0]['_id'], { $set: { timeStamp: time - 600000, countA: countA, countNA: countNA } }, { new: true }).then(
                        docs => res.send(docs),
                        err => res.send(err)
                    );
                }
            } else {
                var ins = Intrested({
                    countA: 0,
                    countNA: 0
                });
                ins.save();
            }


        });
        // res.send(req.body);
    });
    app.get('/api/stats', (req, res) => {
        console.log(typeof(+req.query.no));
        Intrested.find().sort({"_id": -1}).limit(+req.query.no).then(
            (docs) => {console.log(docs.length);res.send(docs)},
            (err) => res.send(err)
        );
    });
    app.get('/api/run', (req, res) => {
        runShell();
        res.send('Go to /dashboard');
    });
    app.get('/api/faceData', (req, res) =>{
        var data = JSON.parse(fs.readFileSync(__dirname + '/../4forces.json'));
        res.send(data);
    });
    app.get('/api/attentive', (req, res) => {
        var data = JSON.parse(fs.readFileSync(__dirname + '/../4forces2.json'));
        res.send(data);
    });
    app.get('/api/people', (req, res) => {
        var data = JSON.parse(fs.readFileSync(__dirname + '/../4forces3.json'));
        res.send(data);
        Intrested.find().sort({"_id": -1}).limit(1).then( (post) => {
            if (post[0]) {
                const now = Date.now();
                var t = ((now - +post[0].timeStamp)/600000);
                console.log(t);
                if(t < 1) {
                    countA = data.len;
                    countNA = 0;
                    console.log('here');
                    Intrested.findByIdAndUpdate(post[0]['_id'], {$set: {timeStamp: post[0].timeStamp, countA: countA, countNA: countNA}},{new : true} ).then(
                    );
                } else {
                    var time = +post[0].timeStamp;
                    while (t > 1) {
                        console.log('here2');
                        var i = Intrested({
                            timeStamp: time + 600000,
                            countA: 0,
                            countNA: 0
                        });
                        i.save();
                        time = time + 600000;
                        t = t - 1;
                    }
                    countA = data.len;
                    countNA = 0;
                    Intrested.findByIdAndUpdate(post[0]['_id'], {$set: {timeStamp: time - 600000, countA: countA, countNA: countNA}},{new : true} ).then(
                    );
                }
            } else {
               var ins = Intrested({
                countA: 0,
                countNA: 0
                });
                ins.save(); 
            }
            
            
    });
    });
}
module.exports = {
    routes
}
// var i = new Intrested({count: 1});
        // var ui = new UnIntrested({count: 2});
        // i.save();
        // ui.save();