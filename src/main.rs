use rand::{prelude::ThreadRng, Rng};
use std::time::Instant;

const NUM_INPUT: usize = 4;
const NUM_OUTPUT: usize = 3;

fn main() {
    const NUM_DATA_ITEMS: usize = 500;
    const MAX_EPOCHS: u32 = 10000;
    const LEARN_RATE: f64 = 0.001;
    const MOMENTUM: f64 = 0.01;

    // bench();

    let start = Instant::now();
    println!("\nBegin deep net training demo \n");

    println!(
        "Generating {:?}  artificial training data items ",
        NUM_DATA_ITEMS
    );

    let train_data = make_data(NUM_DATA_ITEMS);

    println!("\nDone. Training data is: ");
    show_matrix(&train_data, 3, 2, true);

    println!("\nCreating a 4-(10,10,10)-3 deep neural network (tanh & softmax) \n");
    let mut dn = DeepNet::new(NUM_INPUT, NUM_OUTPUT);
    //dn.Dump();

    println!("Setting maxEpochs = {:?} ", MAX_EPOCHS);
    println!("Setting learnRate = {:?} ", LEARN_RATE);
    println!("Setting momentumm = {:?} ", MOMENTUM);
    println!("\nStart training using back-prop with mean squared error \n");
    //let _wts: Vec<f64> =
    dn.train(&train_data, MAX_EPOCHS, LEARN_RATE, MOMENTUM, 10); // show error every maxEpochs / 10
    println!("Training complete \n");

    let train_error = dn.error(&train_data, false);
    let train_acc = dn.accuracy(&train_data, false);
    println!("Final model MS error = {:?} ", train_error);
    println!("Final model accuracy = {:?} ", train_acc);

    println!("\nEnd demo ");
    //Console.ReadLine();
    let duration = start.elapsed();
    println!("Time elapsed is: {:?}", duration);
}

/*
const N: usize = 50000000;

fn bench() {
    let mut arr = [0usize; N];
    let mut vec = vec![0usize; N];

    let now = Instant::now();
    for i in 0..N {
        arr[i] = i;
    }
    println!("Finished arr in: {}", now.elapsed().as_millis());

    let now = Instant::now();
    for i in 0..N {
        vec[i] = i;
    }
    println!("Finished vec in: {}", now.elapsed().as_millis());

    let now = Instant::now();
    for (i, elem) in vec.iter_mut().enumerate() {
        *elem = i;
    }
    println!("Finished vec iter in: {}", now.elapsed().as_millis());
}

*/

fn show_matrix(matrix: &[Vec<f64>], num_rows: usize, decimals: usize, indices: bool) {
    let len = matrix.len().to_string().len();
    for i in 0..num_rows {
        if indices {
            print!("[{:>len$}]  ", i.to_string());
        }
        for j in 0..matrix[i].len() {
            let v: f64 = matrix[i][j];
            if v >= 0.0 {
                print!(" "); // '+'
            }
            print!("{:.decimals$}   ", v);
        }
        println!();
    }

    if num_rows < matrix.len() {
        println!(". . .");
        let last_row = matrix.len() - 1;
        if indices {
            print!("[{:>len$}]  ", last_row.to_string());
        }
        for j in 0..matrix[last_row].len() {
            let v = matrix[last_row][j];
            if v >= 0.0 {
                print!(" "); // '+'
            }
            print!("{:.decimals$}   ", v);
        }
    }
    println!();
}

fn make_data(num_items: usize) -> Vec<Vec<f64>> {
    let mut dn: DeepNet = DeepNet::new(NUM_INPUT, NUM_OUTPUT); // make a DNN generator
    let mut rrnd = rand::thread_rng(); // to make random weights & biases, random input vals

    let wt_lo: f64 = -9.0;
    let wt_hi: f64 = 9.0;

    let nw = dn.num_weights(NUM_INPUT, NUM_OUTPUT);
    let mut wts: Vec<f64> = vec![0.0; nw];

    for i in 0..nw {
        wts[i] = (wt_hi - wt_lo) * rrnd.gen::<f64>() + wt_lo;
    }
    dn.set_weights(&wts);

    let mut result: Vec<Vec<f64>> = vec![vec![0.0; NUM_INPUT + NUM_OUTPUT]; num_items]; // make the result matrix holder

    let in_lo = -4.0; // pseudo-Gaussian scaling
    let in_hi = 4.0;

    for r in 0..num_items {
        let mut inputs: Vec<f64> = vec![0.0; NUM_INPUT]; // random input values

        for i in 0..NUM_INPUT {
            inputs[i] = (in_hi - in_lo) * rrnd.gen::<f64>() + in_lo;
        }

        //ShowVector(inputs, 2);

        let probs = dn.compute_outputs(&inputs); // compute the outputs (as softmax probs) like [0.10, 0.15, 0.55, 0.20]

        //dn.Dump();
        //Console.ReadLine();
        //ShowVector(probs, 4);

        let outputs = probs_to_classes(&probs); // convert to outputs like [0, 0, 1, 0]

        let mut c = 0;
        for i in 0..NUM_INPUT {
            result[r][c] = inputs[i];
            c += 1;
        }
        for i in 0..NUM_OUTPUT {
            result[r][c] = outputs[i];
            c += 1;
        }
        //Console.WriteLine("");
    }

    result
}

fn probs_to_classes(probs: &Vec<f64>) -> Vec<f64> {
    let mut result: Vec<f64> = vec![0.0; probs.len()];
    let idx: usize = max_index(probs);
    result[idx] = 1.0;
    result
}

fn max_index(probs: &[f64]) -> usize {
    let mut max_idx = 0;
    let mut max_val = probs[0];

    for i in 0..probs.len() {
        if probs[i] > max_val {
            max_val = probs[i];
            max_idx = i;
        }
    }
    max_idx
}

pub struct DeepNet {
    pub rnd: ThreadRng,       // weight init and train shuffle
    pub n_input: usize,       // number input nodes
    pub n_hidden: Vec<usize>, // number hidden nodes, each layer
    pub n_output: usize,      // number output nodes
    pub n_layers: usize,      // number hidden node layers

    pub i_nodes: Vec<f64>,      // input nodes
    pub h_nodes: Vec<Vec<f64>>, // array di array jagged
    pub o_nodes: Vec<f64>,

    pub ih_weights: Vec<Vec<f64>>,      // input- 1st hidden
    pub hh_weights: Vec<Vec<Vec<f64>>>, // hidden-hidden
    pub ho_weights: Vec<Vec<f64>>,      // last hidden-output

    pub h_biases: Vec<Vec<f64>>, // hidden node biases
    pub o_biases: Vec<f64>,      // output node biases
    pub ih_gradient00: f64,      // one gradient to monitor
}

impl DeepNet {
    fn new(n_input: usize, n_output: usize) -> Self {
        let hidden: Vec<usize> = vec![10, 10, 10];
        let mut dn: DeepNet = DeepNet {
            n_input,
            n_output,
            n_layers: hidden.len(),
            ih_gradient00: 0.0,
            rnd: rand::thread_rng(),
            i_nodes: vec![0.0; n_input],
            o_nodes: vec![0.0; n_output],
            o_biases: vec![0.0; n_output],
            h_nodes: vec![vec![0.0; hidden[0]]; n_output],
            h_biases: vec![vec![0.0; hidden[0]]; n_output],
            ih_weights: vec![vec![0.0; hidden[0]]; n_input],
            ho_weights: vec![vec![0.0; n_output]; hidden[hidden.len() - 1]],
            hh_weights: vec![vec![vec![0.0; hidden[2]]; hidden[1]]; hidden.len() - 1],
            n_hidden: hidden,
        };

        // make wts
        let lo = -0.10;
        let hi = 0.10;
        let num_wts = dn.num_weights(n_input, n_output);
        let mut wts: Vec<f64> = vec![0.0; num_wts];

        for i in 0..num_wts {
            wts[i] = (hi - lo) * dn.rnd.gen::<f64>() + lo;
        }
        dn.set_weights(&wts);
        dn
    }

    fn num_weights(&self, n_input: usize, n_output: usize) -> usize {
        // total num weights and biases
        let ih_wts = n_input * self.n_hidden[0];

        let mut hh_wts = 0;

        for j in 0..self.n_hidden.len() - 1 {
            hh_wts += self.n_hidden[j] * self.n_hidden[j + 1];
        }

        let ho_wts = self.n_hidden[self.n_hidden.len() - 1] * n_output;

        let mut hbs = 0;

        for i in 0..self.n_hidden.len() {
            hbs += self.n_hidden[i];
        }
        let obs = n_output;

        let nw = ih_wts + hh_wts + ho_wts + hbs + obs;
        return nw;
    }

    fn get_weights(&mut self) -> Vec<f64> {
        // order: ihweights -> hhWeights[] -> hoWeights -> hBiases[] -> oBiases
        let nw = self.num_weights(self.n_input, self.n_output); // total num wts + biases
        let mut result = vec![0.0; nw];

        let mut ptr = 0; // pointer into result[]

        for i in 0..self.n_input {
            // input node
            for j in 0..self.h_nodes[0].len() {
                // 1st hidden layer nodes
                result[ptr] = self.ih_weights[i][j];
                ptr += 1;
            }
        }

        for h in 0..self.n_layers - 1 {
            // not last h layer
            for j in 0..self.n_hidden[h] {
                // from node
                for jj in 0..self.n_hidden[h + 1] {
                    // to node
                    result[ptr] = self.hh_weights[h][j][jj];
                    ptr += 1;
                }
            }
        }

        let hi = self.n_layers - 1; // if 3 hidden layers (0,1,2) last is 3-1 = [2]
        for j in 0..self.n_hidden[hi] {
            for k in 0..self.n_output {
                result[ptr] = self.ho_weights[j][k];
                ptr += 1;
            }
        }

        for h in 0..self.n_layers {
            // hidden node biases
            for j in 0..self.n_hidden[h] {
                result[ptr] = self.h_biases[h][j];
                ptr += 1;
            }
        }
        for k in 0..self.n_output {
            result[ptr] = self.o_biases[k];
            ptr += 1;
        }

        return result;
    }

    fn set_weights(&mut self, wts: &Vec<f64>) {
        // order: ihweights - hhWeights[] - hoWeights - hBiases[] - oBiases
        let nw = self.num_weights(self.n_input, self.n_output); // total num wts + biases
        if wts.len() != nw {
            panic!("Bad wts[] length in set_weights()")
        }
        let mut ptr = 0; // pointer into wts[]

        for i in 0..self.n_input {
            // input node
            for j in 0..self.h_nodes[0].len() {
                self.ih_weights[i][j] = wts[ptr];
                ptr += 1;
            }
        }
        for h in 0..self.n_layers - 1 {
            // not last h layers
            for j in 0..self.n_hidden[h] {
                // from node
                for jj in 0..self.n_hidden[h + 1] {
                    // to node
                    self.hh_weights[h][j][jj] = wts[ptr];
                    ptr += 1;
                }
            }
        }
        let hi = self.n_layers - 1; // if 3 hidden layers (0,1,2) last is 3-1 = [2]
        for j in 0..self.n_hidden[hi] {
            for k in 0..self.n_output {
                self.ho_weights[j][k] = wts[ptr];
                ptr += 1;
            }
        }

        for h in 0..self.n_layers {
            // hidden node biases
            for j in 0..self.n_hidden[h] {
                self.h_biases[h][j] = wts[ptr];
                ptr += 1;
            }
        }
        for k in 0..self.n_output {
            self.o_biases[k] = wts[ptr];
            ptr += 1;
        }
    }

    fn compute_outputs(&mut self, x_values: &[f64]) -> Vec<f64> {
        // 'xValues' might have class label or not
        // copy vals into iNodes

        for i in 0..self.n_input {
            // possible trunc
            self.i_nodes[i] = x_values[i];
        }

        // zero-out all hNodes, oNodes
        for h in 0..self.n_layers {
            for j in 0..self.n_hidden[h] {
                self.h_nodes[h][j] = 0.0;
            }
        }

        for k in 0..self.n_output {
            self.o_nodes[k] = 0.0;
        }

        // input to 1st hid layer
        for j in 0..self.n_hidden[0] {
            // each hidden node, 1st layer
            for i in 0..self.n_input {
                self.h_nodes[0][j] += self.ih_weights[i][j] * self.i_nodes[i];
            }
            // add the bias
            //hNodes[0][j] += hBiases[0][j];
            //// apply activation
            //hNodes[0][j] = Math.Tanh(hNodes[0][j]);

            //hNodes[0][j] = Math.Tanh(hNodes[0][j] + hBiases[0][j]);
            self.h_nodes[0][j] = DeepNet::my_tanh(self.h_nodes[0][j] + self.h_biases[0][j]);
        }

        // each remaining hidden node
        for h in 1..self.n_layers {
            for j in 0..self.n_hidden[h] {
                // 'to index'
                for jj in 0..self.n_hidden[h - 1] {
                    // 'from index'
                    self.h_nodes[h][j] += self.hh_weights[h - 1][jj][j] * self.h_nodes[h - 1][jj];
                }
                //hNodes[h][j] += hBiases[h][j];  // add bias value
                //hNodes[h][j] =  Math.Tanh(hNodes[h][j]);  // apply activation

                //hNodes[h][j] = Math.Tanh(hNodes[h][j] + hBiases[h][j]);  // apply activation
                self.h_nodes[h][j] = DeepNet::my_tanh(self.h_nodes[h][j] + self.h_biases[h][j]);
                // apply activation
            }
        }

        // compute ouput node values
        for k in 0..self.n_output {
            for j in 0..self.n_hidden[self.n_layers - 1] {
                self.o_nodes[k] += self.ho_weights[j][k] * self.h_nodes[self.n_layers - 1][j];
            }
            self.o_nodes[k] += self.o_biases[k]; // add bias value
                                                 //Console.WriteLine("Pre-softmax output node [" + k + "] value = " + oNodes[k].ToString("F4"));
        }

        let ret_result: Vec<f64> = DeepNet::soft_max(&self.o_nodes); // softmax activation all oNodes

        for k in 0..self.n_output {
            self.o_nodes[k] = ret_result[k];
        }
        //return ret_result; // calling convenience
        ret_result
    } // compute_outputs

    fn my_tanh(x: f64) -> f64 {
        if x < -20.0 {
            return -1.0;
        }
        // approximation is correct to 30 decimals
        else if x > 20.0 {
            return 1.0;
        } else {
            return x.tanh();
        }
    }

    fn soft_max(o_sums: &Vec<f64>) -> Vec<f64> {
        // does all output nodes at once so scale
        // doesn't have to be re-computed each time.
        // possible overflow . . . use max trick

        let mut sum = 0.0;
        for i in 0..o_sums.len() {
            sum += o_sums[i].exp();
        }
        let mut result: Vec<f64> = vec![0.0; o_sums.len()];

        for i in 0..o_sums.len() {
            result[i] = o_sums[i].exp() / sum;
        }
        return result; // now scaled so that xi sum to 1.0
    }

    fn show_vector(vector: &Vec<f64>, dec: usize) {
        for i in vector {
            print!("{:.dec$}   ", i);
        }
        println!();
    }

    fn accuracy(&mut self, data: &[Vec<f64>], verbose: bool) -> f64 {
        // percentage correct using winner-takes all
        let mut num_correct: f64 = 0.0;
        let mut num_wrong: f64 = 0.0;

        for i in data {
            let x_values = Vec::from_iter(i[0..self.n_input].iter().cloned()); // get x-values
            let t_values = Vec::from_iter(
                i[self.n_input..(self.n_input + self.n_output)]
                    .iter()
                    .cloned(),
            ); // get target values
            let y_values = self.compute_outputs(&x_values); // outputs using current weights

            if verbose {
                DeepNet::show_vector(&y_values, 4);
                DeepNet::show_vector(&t_values, 4);
                println!();
                //Console.ReadLine();
            }

            let max_index0: usize = max_index(&y_values); // which cell in yValues has largest value?
            let t_max_index: usize = max_index(&t_values);

            if max_index0 == t_max_index {
                num_correct += 1.0;
            } else {
                num_wrong += 1.0;
            }
        }
        return (num_correct * 1.0) / (num_correct + num_wrong);
    }

    fn error(&mut self, data: &Vec<Vec<f64>>, verbose: bool) -> f64 {
        // mean squared error using current weights & biases
        let mut sum_squared_error = 0.0;

        // walk thru each training case. looks like (6.9 3.2 5.7 2.3) (0 0 1)
        for i in data {
            let x_values = Vec::from_iter(i[0..self.n_input].iter().cloned());
            let t_values = Vec::from_iter(
                i[self.n_input..(self.n_input + self.n_output)]
                    .iter()
                    .cloned(),
            ); // get target values
            let y_values = self.compute_outputs(&x_values); // outputs using current weights

            if verbose {
                DeepNet::show_vector(&y_values, 4);
                DeepNet::show_vector(&t_values, 4);
                println!();
                //Console.ReadLine();
            }

            for j in 0..self.n_output {
                let err = t_values[j] - y_values[j];
                sum_squared_error += err * err;
            }
        }

        return sum_squared_error / ((data.len() * self.n_output) as f64); // average per item
    } // error

    fn train(
        &mut self,
        train_data: &Vec<Vec<f64>>,
        max_epochs: u32,
        learn_rate: f64,
        momentum: f64,
        show_every: u32,
    ) // -> Vec<f64>
    {
        // no momentum right now
        // each weight (and bias) needs a big_delta. big_delta is just learnRate * "a gradient"
        // so goal is to find "a gradient".
        // the gradient (the term can have several meanings) is "a signal" * "an input"
        // the signal

        // 1. each weight and bias has a 'gradient' (partial dervative)
        let mut ho_grads = vec![vec![0.0; self.n_output]; self.n_hidden[self.n_layers - 1]]; // last_hidden layer - output weights grads
        let mut hh_grads: Vec<Vec<Vec<f64>>> =
            vec![vec![vec![0.0; self.n_hidden[1]]; self.n_hidden[0]]; self.n_layers - 1];

        let mut ih_grads: Vec<Vec<f64>> = vec![vec![0.0; self.n_hidden[0]]; self.n_input]; // input-first_hidden wts gradients
                                                                                           // biases
        let mut ob_grads = vec![0.0; self.n_output]; // output node bias grads
        let mut hb_grads = vec![vec![0.0; self.n_hidden[0]]; self.n_output]; // hidden node bias grads

        // 2. each output node and each hidden node has a 'signal' == gradient without associated input (lower case delta in Wikipedia)
        let mut o_signals = vec![0.0; self.n_output];
        let mut h_signals = vec![vec![0.0; self.n_hidden[0]]; self.n_output];

        // 3. for momentum, each weight and bias needs to store the prev epoch delta
        // the structure for prev deltas is same as for Weights & Biases, which is same as for Grads

        let mut ho_prev_weights_delta =
            vec![vec![0.0; self.n_output]; self.n_hidden[self.n_layers - 1]]; // last_hidden layer - output weights momentum term
        let mut hh_prev_weights_delta: Vec<Vec<Vec<f64>>> =
            vec![vec![vec![0.0; self.n_hidden[1]]; self.n_hidden[0]]; self.n_layers - 1];

        let mut ih_prev_weights_delta = vec![vec![0.0; self.n_hidden[0]]; self.n_input]; // input-first_hidden wts gradients
        let mut o_prev_biases_delta = vec![0.0; self.n_output]; // output node bias prev deltas
        let mut h_prev_biases_delta =
            vec![vec![0.0; self.n_hidden[self.n_layers - 1]]; self.n_output]; // hidden node bias prev deltas

        let mut epoch = 0;

        let mut sequence = vec![0; train_data.len()];

        for i in 0..sequence.len() {
            sequence[i] = i;
        }

        let err_interval = max_epochs / show_every; // interval to check & display  Error

        while epoch < max_epochs {
            epoch += 1;
            if epoch % err_interval == 0 && epoch < max_epochs
            // display curr MSE
            {
                let train_err = self.error(train_data, false); // using curr weights & biases
                let train_acc = self.accuracy(train_data, false);
                print!("epoch = {:?}  MS error = {:.4} ", epoch, train_err);
                println!("  accuracy = {:.4} ", train_acc);
                println!(
                    "input-to-hidden [0][0] gradient = {:.12}",
                    self.ih_gradient00
                );
                println!();
                //this.Dump();
                // Console.ReadLine();
            }

            DeepNet::shuffle(&mut sequence); // must visit each training data in random order in stochastic GD

            for ii in 0..train_data.len() {
                // each train data item
                let idx = sequence[ii]; // idx points to a data item

                let x_values = Vec::from_iter(train_data[idx][0..self.n_input].iter().cloned()); // get x-values
                let t_values = Vec::from_iter(
                    train_data[idx][self.n_input..(self.n_input + self.n_output)]
                        .iter()
                        .cloned(),
                ); // get target values

                self.compute_outputs(&x_values); // copy xValues in, compute outputs using curr weights & biases, ignore return

                // must compute signals from right-to-left
                // weights and bias gradients can be computed left-to-right
                // weights and bias gradients can be updated left-to-right

                // x. compute output node signals (assumes softmax) depends on target values to the right
                for k in 0..self.n_output {
                    let error_signal = t_values[k] - self.o_nodes[k]; // Wikipedia uses (o-t)
                    let derivative = (1.0 - self.o_nodes[k]) * self.o_nodes[k]; // for softmax (same as log-sigmoid) with MSE
                                                                                //derivative = 1.0;  // for softmax with cross-entropy
                    o_signals[k] = error_signal * derivative; // we'll use this for ho-gradient and hSignals
                }

                // x. compute signals for last hidden layer (depends on oNodes values to the right)
                let last_layer = self.n_layers - 1;

                for j in 0..self.n_hidden[last_layer] {
                    let mut sum: f64 = 0.0; // need sums of output signals times hidden-to-output weights
                    let derivative =
                        (1.0 + self.h_nodes[last_layer][j]) * (1.0 - self.h_nodes[last_layer][j]); // for tanh!
                    for k in 0..self.n_output {
                        sum += o_signals[k] * self.ho_weights[j][k]; // represents error signal
                    }
                    h_signals[last_layer][j] = derivative * sum;
                }

                // x. compute signals for all the non-last layer hidden nodes (depends on layer to the right)
                for h in (0..(self.n_layers - 1)).rev() {
                    // each hidden layer, right-to-left
                    for j in 0..self.n_hidden[h] {
                        let mut sum = 0.0; // need sums of output signals times hidden-to-output weights
                                           // each node
                        let derivative = (1.0 + self.h_nodes[h][j]) * (1.0 - self.h_nodes[h][j]); // for tanh
                                                                                                  // derivative = hNodes[h][j];
                        for jj in 0..self.n_hidden[h + 1] {
                            // layer to right of curr layer
                            sum += h_signals[h + 1][jj] * self.hh_weights[h][j][jj];
                        }
                        h_signals[h][j] = derivative * sum;
                    } // j
                } // h

                // at this point, all hidden and output node signals have been computed
                // calculate gradients left-to-right

                // x. compute input-to-hidden weights gradients using iNodes & hSignal[0]
                for i in 0..self.n_input {
                    for j in 0..self.n_hidden[0] {
                        ih_grads[i][j] = self.i_nodes[i] * h_signals[0][j]; // "from" input & "to" signal
                    }
                }

                // save the special monitored ihGradient00
                self.ih_gradient00 = ih_grads[0][0];

                // x. compute hidden-to-hidden gradients
                for h in 0..self.n_layers - 1 {
                    for j in 0..self.n_hidden[h] {
                        for jj in 0..self.n_hidden[h + 1] {
                            hh_grads[h][j][jj] = self.h_nodes[h][j] * h_signals[h + 1][jj];
                        }
                    }
                }

                // x. compute hidden-to-output gradients
                for j in 0..self.n_hidden[last_layer] {
                    /* for (j, item) in ho_grads
                        .iter_mut()
                        .enumerate()
                        .take(self.n_hidden[last_layer])
                    {*/
                    for k in 0..self.n_output {
                        //for (k, elem) in o_signals.iter().enumerate().take(self.n_output) {
                        ho_grads[j][k] = self.h_nodes[last_layer][j] * o_signals[k];
                        //item[k] = self.h_nodes[last_layer][j] * *elem;
                        // from last hidden, to oSignals
                    }
                }

                // compute bias gradients
                // a bias is like a weight on the left/before
                // so there's a dummy input of 1.0 and we use the signal of the 'current' layer

                // x. compute all hidden bias gradients
                // a gradient needs the "left/from" input and the "right/to" signal
                // for biases we use a dummy 1.0 input

                for h in 0..self.n_layers {
                    for j in 0..self.n_hidden[h] {
                        hb_grads[h][j] = 1.0 * h_signals[h][j];
                    }
                }

                // x. output bias gradients
                for k in 0..self.n_output {
                    ob_grads[k] = 1.0 * o_signals[k];
                }

                // at this point all signals have been computed and all gradients have been computed
                // so can use gradients to update all weights and biases.
                // save each delta for the momentum

                // x. update input-to-first_hidden weights using ihWeights & ihGrads
                for i in 0..self.n_input {
                    for j in 0..self.n_hidden[0] {
                        let delta = ih_grads[i][j] * learn_rate;
                        //ihWeights[i][j] += delta;
                        //ihWeights[i][j] += ihPrevWeightsDelta[i][j] * momentum;

                        self.ih_weights[i][j] += delta + ih_prev_weights_delta[i][j] * momentum;

                        ih_prev_weights_delta[i][j] = delta;
                    }
                }
                // other hidden-to-hidden weights using hhWeights & hhGrads
                for h in 0..self.n_layers - 1 {
                    for j in 0..self.n_hidden[h] {
                        for jj in 0..self.n_hidden[h + 1] {
                            let delta = hh_grads[h][j][jj] * learn_rate;
                            //hhWeights[h][j][jj] += delta;
                            //hhWeights[h][j][jj] += hhPrevWeightsDelta[h][j][jj] * momentum;

                            self.hh_weights[h][j][jj] +=
                                delta + hh_prev_weights_delta[h][j][jj] * momentum;
                            hh_prev_weights_delta[h][j][jj] = delta;
                        }
                    }
                }

                // update hidden-to-output weights using hoWeights & hoGrads
                for j in 0..self.n_hidden[last_layer] {
                    for k in 0..self.n_output {
                        let delta = ho_grads[j][k] * learn_rate;
                        //hoWeights[j][k] += delta;
                        //hoWeights[j][k] += hoPrevWeightsDelta[j][k] * momentum;

                        self.ho_weights[j][k] += delta + ho_prev_weights_delta[j][k] * momentum;
                        ho_prev_weights_delta[j][k] = delta;
                    }
                }

                // update hidden biases using hBiases & hbGrads
                for h in 0..self.n_layers {
                    for j in 0..self.n_hidden[h] {
                        let delta = hb_grads[h][j] * learn_rate;
                        //hBiases[h][j] += delta;
                        //hBiases[h][j] += hPrevBiasesDelta[h][j] * momentum;

                        self.h_biases[h][j] += delta + h_prev_biases_delta[h][j] * momentum;
                        h_prev_biases_delta[h][j] = delta;
                    }
                }

                // update output biases using oBiases & obGrads
                for k in 0..self.n_output {
                    let delta = ob_grads[k] * learn_rate;
                    //oBiases[k] += delta;
                    //oBiases[k] += oPrevBiasesDelta[k] * momentum;

                    self.o_biases[k] += delta + o_prev_biases_delta[k] * momentum;
                    o_prev_biases_delta[k] = delta;
                }

                // Whew!
            } // for each train data item
        } // while

        let _best_wts = self.get_weights();
        //return best_wts;
    } // Train

    fn shuffle(sequence: &mut [usize]) // instance method
    {
        let mut rnd = rand::thread_rng();
        for i in 0..sequence.len() {
            let r = rnd.gen_range(i..sequence.len());
            /*let tmp = sequence[r];
            sequence[r] = sequence[i];
            sequence[i] = tmp;*/
            sequence.swap(r, i);
        }
    } // Shuffle
}
