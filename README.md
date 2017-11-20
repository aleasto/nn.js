# nn.js

> A simple feed-forward Neural Network implementation in ES6 JavaScript 

[![NPM Version][npm-image]][npm-url]

## Install

```bash
npm install nn.js
```

## Usage
```javascript
// Import the NeuralNetwork class
const NeuralNetwork = require("nn.js");

// Create a new Neural Network.

// The first 3+ arguments are the number of neurons for each layer. 
//  It supports multiple Hidden layers.

// The last argument specifies which activation function to use.
//  Currently supports "logistic", "tanh", "relu".
let nn = new NeuralNetwork(10,7,5,2,"relu");

// Set up your input and target data as a batch of arrays
let inputs = [
    [0,0,0,0,0,1,1,1,1,1],  // First input*
    [1,1,1,1,1,0,0,0,0,0]   // Second input**
];
let targets = [
    [0,1],                  // First target*
    [1,0]                   // Second target**
];

// Feed the network with an input, returns the output
console.log("Result: " + nn.eval(inputs[0]));

// Call the train function, passing:
//  inputs, targets, learning rate, (optional) number of epochs, (optional) minimum MSE before stopping
//  The number of epochs HAS PRIORITY on the conditions. Meaning that if you specify both, it will only run for specified epochs.
nn.train(inputs, targets, 0.01, 40, 0.05);

//  This instead will truly run until MSE < 0.05
nn.train(inputs, targets, 0.01, null, 0.05)

// You can also call the trainAsync function, to have it train asynchronously
nn.trainAsync(inputs, targets, 0.01, null, 0.05)
// You can stop the async training at any point by calling
nn.stopTraining();

// Save the current weight and bias values
let data = nn.save();
JSON.stringify(data);

// Or load them
nn.load(data);

// K.I.S.S!
```

## License

ISC License

Copyright (c) 2017, Alessandro Astone

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.


[npm-image]: https://img.shields.io/npm/v/nn.js.svg
[npm-url]: https://npmjs.org/package/nn.js