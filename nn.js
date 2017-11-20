class NeuralNetwork {
    constructor(){
        if(arguments.length<4){
            throw "Invalid arguments";
        }

        // Get numbers
        let inputNeurons = arguments[0];
        let hiddenNeuronsArray = new Array(arguments.length-3);
        for(let i=1; i<arguments.length-2; i++){
            hiddenNeuronsArray[i-1] = arguments[i];
        }
        let outputNeurons = arguments[arguments.length-2];

        // Create layers
        this.inputLayer = new Layer(inputNeurons);
        this.hiddenLayers = new Array(hiddenNeuronsArray.length);
        this.hiddenLayers[0] = new Layer(hiddenNeuronsArray[0], this.inputLayer);
        for(let i=1; i<this.hiddenLayers.length; i++){
            this.hiddenLayers[i] = new Layer(hiddenNeuronsArray[i], this.hiddenLayers[i-1]);
        }
        this.outputLayer = new Layer(outputNeurons, this.hiddenLayers[this.hiddenLayers.length-1]);

        // Initializa weight and bias values
        for(let i=0; i<this.hiddenLayers.length; i++){
            this.hiddenLayers[i].initWeights();
            this.hiddenLayers[i].initBiases();
        }
        this.outputLayer.initWeights();
        this.outputLayer.initBiases();

        // Set activation function
        let f;
        switch(arguments[arguments.length-1]){
            case "logistic":
                f = logistic;
                break;
            case "tanh":
                f = tanh;
                break;
            case "relu":
                f = relu;
                break;

            default:
                throw "Activation function not defined";
        }
        this.activation = f.activation;
        this.derivative = f.derivative;
    }

    /**
     * Runs feed-forward
     * @param {*} inputValues 
     * @returns {Number[]} - Output values
     */
    eval(inputValues){
        if(inputValues.length != this.inputLayer.neurons.length){
            throw "Input values array has a different size than the Neural Net's input layer";
        }
        return this.inputLayer.feed(inputValues, this.activation);
    }

    /**
     * Calculate Mean Squared Error on the current 
     * @param {Number[]} inputValues - Input values
     * @param {Number[]} targetValues - Target values
     * @returns {Number} - Mean Squared Error
     */
    mse(inputValues, targetValues){
        if(inputValues.length != this.inputLayer.neurons.length || targetValues.length != this.outputLayer.neurons.length){
            throw "Wrong array sizes"
        }

        this.eval(inputValues);
        let sum = 0;
        for(let i=0; i<this.outputLayer.neurons.length; i++){
            sum += Math.pow(targetValues[i] - this.outputLayer.neurons[i], 2);
        }
        return sum/2;
    }

    /**
     * Runs backpropagation on the current output values
     * @param {Number[]} targetValues - Array of target values
     * @param {Number} learnRate - Small positive number that defines how quickly to train
     */
    backprop(targetValues, learnRate){
        if(targetValues.length != this.outputLayer.neurons.length){
            throw "Target values array has a different size than the Neural Net's output layer";
        }

        /* Eval output layer deltas */
        for(let k=0; k<this.outputLayer.neurons.length; k++){
            this.outputLayer.deltas[k] = -1 * (targetValues[k] - this.outputLayer.neurons[k]) * this.derivative(this.outputLayer.neurons[k]);
        }
        
        /* Update output weights and biases */
        for(let j=0; j<this.outputLayer.prev.neurons.length; j++){
            for(let k=0; k<this.outputLayer.neurons.length; k++){
                this.outputLayer.weights[j][k] += - learnRate * this.outputLayer.deltas[k] * this.outputLayer.prev.neurons[j];
            }
        }
        for(let k=0; k<this.outputLayer.neurons.length; k++){
            this.outputLayer.biases[k] += - learnRate * this.outputLayer.deltas[k];
        }
        /* Chain */
        this.outputLayer.prev.back(learnRate, this.derivative);
    }

    /**
     * Train the neural network
     * @param {Number[][]} inputs - Batch of input values
     * @param {Number[][]} targets - Batch of target values
     * @param {Number} learnRate - Small positive number that defines how quickly to train
     * @param {Number} [epochs = Infinity] - Number of iterations of the whole training set
     * @param {Number} [minMSE = -1] - Minimum value of MeanSquaredError on training data
     */
    train(inputs, targets, learnRate, epochs, minMSE){
        if(!epochs && !minMSE)
                throw "At least one valid finish condition must be specified";

        if(inputs.length != targets.length){
            throw "Inputs should be as many as targets";
        }
        
        if(!epochs)
            epochs = Infinity;
        if(!minMSE)
            minMSE = -1;

        let i=0;
        do{
            for(let j=0; j<inputs.length; j++){
                this.eval(inputs[j]);
                this.backprop(targets[j], learnRate);
            }
            i++;
        }
        while(i<epochs && this.mse(inputs[0], targets[0]) > minMSE );
    }

    /**
     * Train the neural network asynchronously
     * @param {Number[][]} inputs - Batch of input values
     * @param {Number[][]} targets - Batch of target values
     * @param {Number} learnRate - Small positive number that defines how quickly to train
     * @param {Number} [epochs = Infinity] - Number of iterations of the whole training set
     * @param {Number} [minMSE = -1] - Minimum value of MeanSquaredError on training data
     */
    async trainAsync(inputs, targets, learnRate, epochs, minMSE){
        if(inputs.length != targets.length){
            throw "Inputs should be as many as targets";
        }

        if(!epochs)
            epochs = Infinity;
        if(!minMSE)
            minMSE = -1;

        this.stop = false;
        let i=0;
        do{
            for(let j=0; j<inputs.length; j++){
                this.eval(inputs[j]);
                this.backprop(targets[j], learnRate);
            }
            await new Promise(resolve => setImmediate(resolve));
            i++;
        }
        while(!this.stop && i<epochs && this.mse(inputs[0], targets[0]) > minMSE );
    }

    stopTraining(){
        this.stop = true;
    }

    print(filename){
        let fs = require('fs');
        let data = new Array(this.hiddenLayers.length+1);
        let i=0;
        for(; i<data.length; i++)
            data[i] = { "weights" : [], "biases" : [] }

        for(i=0; i<this.hiddenLayers.length; i++){
            data[i].weights = this.hiddenLayers[i].weights;
            data[i].biases = this.hiddenLayers[i].biases;
        }
        data[i].weights = this.outputLayer.weights;
        data[i].biases = this.outputLayer.biases;

        try{
            fs.writeFileSync(filename, JSON.stringify(data), "utf8");
        }
        catch (err){
            throw "Error while saving file: " + err;
        }
    }

    load(filename){
        try{
            let fs = require('fs');
            let json =fs.readFileSync(filename, "utf8");
            let data = JSON.parse(json);
            // Check the number of layers
            if(data.length != this.hiddenLayers.length+1)
                throw "JSON data does not match this Neural Network size";
            
            // Check each matrix size
            let i=0;
            for(; i<this.hiddenLayers.length; i++){
                if(this.hiddenLayers[i].weights.length != data[i].weights.length || 
                    this.hiddenLayers[i].weights[0].length != data[i].weights[0].length ||
                    this.hiddenLayers[i].biases.length != data[i].biases.length)
                    throw "JSON data does not match this Neural Network size";
            }
            if(this.outputLayer.weights.length != data[i].weights.length ||
                this.outputLayer.weights[0].length != data[i].weights[0].length ||
                this.outputLayer.biases.length != data[i].biases.length)
                throw "JSON data does not match this Neural Network size";

            // All good, procede.
            for(i=0; i<this.hiddenLayers.length; i++){
                this.hiddenLayers[i].loadWeights(data[i].weights);
                this.hiddenLayers[i].loadBiases(data[i].biases);
            }
            this.outputLayer.loadWeights(data[i].weights);
            this.outputLayer.loadBiases(data[i].biases);
        }
        catch (err){
            throw "Error while reading file: " + err;
        }
    }
}

class Layer {
    constructor(n, prev){
        this.neurons = new Array(n);
        if(prev){
            this.prev = prev;
            prev.next = this;

            this.weights = matrix(this.prev.neurons.length, this.neurons.length);
            this.biases = new Array(this.neurons.length);
            this.deltas = new Array(this.neurons.length);
        }
    }

    initWeights(){
        let max = Math.sqrt(2/this.prev.neurons.length);
        for(let i=0; i<this.prev.neurons.length; i++){
            for(let j=0; j<this.neurons.length; j++){
                this.weights[i][j] = Math.random()*max;
            }
        }
    }

    initBiases(){
        for(let i=0; i<this.biases.length; i++){
            this.biases[i] = 0.1;
        }
    }

    feed(x, activation){
        this.neurons = x;
        if(!this.next)   return this.neurons;

        let outputs = new Array(this.next.neurons.length);
        for(let j=0; j<this.next.neurons.length; j++){
            let sum = 0;
            for(let i=0; i<this.neurons.length; i++){
                sum += this.neurons[i] * this.next.weights[i][j];
            }
            sum += this.next.biases[j];
            outputs[j] = activation(sum);
        }
        return this.next.feed(outputs, activation);
    }

    back(learnRate, derivative){
        /* Eval hidden layer deltas */
        for(let i=0; i<this.neurons.length; i++){
            let sum = 0;
            for(let j=0; j<this.next.neurons.length; j++){
                sum += this.next.deltas[j] * this.next.weights[i][j];
            }
            this.deltas[i] = sum * derivative(this.neurons[i]);
        }
        
        /* Update hidden weights and biases */
        for(let i=0; i<this.prev.neurons.length; i++){
            for(let j=0; j<this.neurons.length; j++){
                this.weights[i][j] += - learnRate * this.deltas[j] * this.prev.neurons[i];
            }
        }
        for(let j=0; j<this.neurons.length; j++){
            this.biases[j] += - learnRate * this.deltas[j];
        }

        /* Chain */
        if(this.prev.deltas)
            this.prev.back(learnRate, derivative);
    }

    loadWeights(data){
        this.weights = data;
    }

    loadBiases(data){
        this.biases = data;
    }
}

function matrix(a,b){
    let m = new Array(a);
    for(let i=0; i<a; i++){
        m[i] = new Array(b);
    }
    return m;
}

const logistic = {
    activation: (x) => 1/(1+Math.exp(-x)),
    derivative: (logistic_x) => logistic_x * (1 - logistic_x)
}

const tanh = {
    activation: (x) => Math.tanh(x),
    derivative: (tanh_x) => 1 - Math.pow(tanh_x, 2)
}

const relu = {
    activation: (x) => Math.max(0,x),
    derivative: (relu_x) => relu_x == 0 ? 0 : 1
}

module.exports = NeuralNetwork;