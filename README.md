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
const NeuralNetwork = require("./nn.js");


// Create a new Neural Network.

// The first 3+ arguments are the number of neurons for each layer. 
//  It supports multiple Hidden layers.

// The last argument specifies which activation function to use.
//  Currently supports "logistic", "tanh", "relu".

let nn = new NeuralNetwork(10,7,5,2,"relu");

// Set up your input and target data as a batch of arrays
let inputs = [
    [0,0,0,0,0,1,1,1,1,1],
    [1,1,1,1,1,0,0,0,0,0]
];
let targets = [
    [0,1],
    [1,0]
];

// Call the train function, passing:
//  inputs, targets, learning rate, (optional) number of epochs, (optional) minimum MSE before stopping
nn.train(inputs, targets, 0.01, 40, 0.05);

// You can also call the trainAsync function, to have it train asynchronously
(async function(){
    await nn.trainAsync(inputs, targets, 0.01, 40, 0.05);
    // Call the eval function, to see what the outputs are for a given input
    console.log("Result: " + nn.eval(inputs[0]));
})();

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
[npm-url]: https://npmjs.org/package/nn,js