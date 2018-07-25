const fs = require('fs');;
var imageRoutes = (app) => {
    app.post('/api/images', (req, res) => {
        fs.writeFile(__dirname + '/../FINAL/src/' + 'photo.jpg', req.body.image, {
            encoding: 'base64'
        }, function (err) {
            if (err) console.log(err)
            console.log('File created');
            res.send({
                success: 'success'
            })
        });
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