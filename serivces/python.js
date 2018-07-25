var PythonShell = require('python-shell');

var runShell = () => {
    console.log('here');
    var options = {
        mode: 'text'
    };
    var pyshell = new PythonShell('wordCloud.py', options);
    pyshell.on('message', message => console.log(message))
}
var getEmotion = () => {
    var options = {
        mode: 'text'
    };
    var pyshell = new PythonShell('./FINAL/src/main.py', options);
    pyshell.on('message', message => console.log(message))
}
var getAttentive = () => {
    var options = {
        mode: 'text'
    };
    var pyshell = new PythonShell('./Attention/multi_face.py', options);
    pyshell.on('message', message => console.log(message))
}
var getPeople = () => {
    var options = {
        mode: 'text'
    };
    var pyshell = new PythonShell('./people_count/people_count.py', options);
    pyshell.on('message', message => console.log(message))
}
// runShell();
module.exports = {runShell, getEmotion, getAttentive, getPeople};