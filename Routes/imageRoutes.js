const fs = require('fs');
semaphore = true;
var imageRoutes = (app) => {
    app.post('/api/images', (req, res) => {
        if (semaphore) {
            fs.writeFile(__dirname + '/../FINAL/src/' + '1.jpg', req.body.image, {
                encoding: 'base64'
            }, function (err) {
                if (err) console.log(err)
                console.log('File created1');
                fs.writeFile(__dirname + '/../FINAL/src/' + '3.txt', 'a', (err) => { if(err)  console.log(err)});
                semaphore = false;
                res.send({
                    success: 'success'
                })
            });
        } else {
            fs.writeFile(__dirname + '/../FINAL/src/' + '2.jpg', req.body.image, {
                encoding: 'base64'
            }, function (err) {
                if (err) console.log(err)
                console.log('File created2');
                fs.unlink(__dirname + '/../FINAL/src/' + '3.txt', (err) => {
                    if (err) console.log(err)
                })
                // fs.writeFile(__dirname + '/../FINAL/src/' + '3.txt', 'a');
                semaphore = true;
                res.send({
                    success: 'success'
                })
            });
        }
        // fs.writeFile(__dirname + '/../Attention/' + 'photo.jpg', req.body.image, {
        //     encoding: 'base64'
        // }, function (err) {
        //     if (err) console.log(err);
        // });
        // fs.writeFile(__dirname + '/../people_count/' + 'photo.jpg', req.body.image, {
        //     encoding: 'base64'
        // }, function (err) {
        //     if (err) console.log(err);
        // });
    });
}

module.exports = {
    imageRoutes
}