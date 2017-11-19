const NeuralNetwork = require("./nn.js");

let nn = new NeuralNetwork(10,5,2,"relu");
let inputs = [
    [0,0,0,0,0,1,1,1,1,1],
    [1,1,1,1,1,0,0,0,0,0]
];
let outputs = [
    [0,1],
    [1,0]
];

(async function(){
    await nn.trainAsync(inputs, outputs, 0.01, 40, 0.05);
    console.log("Result: " + nn.eval(inputs[0]));
})();

nn.train(inputs, outputs, 0.01, 40, 0.05);