//use rayon::prelude::*;
//use oorandom;
use std::time::Instant;

const NUM_INPUT: usize = 4;
const NUM_OUTPUT: usize = 3;
const NUM_LAYERS: usize = 3;
const NUM_DATA_ITEMS: usize = 500;
const MAX_EPOCHS: u32 = 10000;
const LEARN_RATE: f64 = 0.001;
const MOMENTUM: f64 = 0.01;

fn main() {
    // bench();

    let start = Instant::now();
    println!("\nBegin deep net training demo \n");

    println!(
        "Generating {:?}  artificial training data items ",
        NUM_DATA_ITEMS
    );

    let train_data = make_data();

    println!("\nDone. Training data is: ");
    show_matrix(&train_data, 3, 2, true);

    println!("\nCreating a 4-(10,10,10)-3 deep neural network (tanh & softmax) \n");
    let mut dn = DeepNet::new();
    //dn.Dump();

    println!("Setting maxEpochs = {:?} ", MAX_EPOCHS);
    println!("Setting learnRate = {:?} ", LEARN_RATE);
    println!("Setting momentumm = {:?} ", MOMENTUM);
    println!("\nStart training using back-prop with mean squared error \n");
    dn.train(&train_data, MAX_EPOCHS, LEARN_RATE, MOMENTUM, 10); // show error every maxEpochs / 10
    println!("Training complete \n");

    println!("Final model MS error = {:?} ", dn.error(&train_data));
    println!("Final model accuracy = {:?} ", dn.accuracy(&train_data));

    println!("\nEnd demo ");

    let duration = start.elapsed();
    println!("Time elapsed is: {:?}", duration);
}

/*
#[inline]
fn my_tanh(x: f64) -> f64 {
    if x < -20.0 {
        -1.0
    }
    // approximation is correct to 30 decimals
    else if x > 20.0 {
        1.0
    } else {
        x.tanh()
    }
}
*/

#[inline]
fn tanh_levien(x: f64) -> f64 {
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;

    let a = x + (0.16489087 * x3) + (0.00985468 * x5);

    a / (1.0 + (a * a)).sqrt()
}

fn show_matrix(
    matrix: &[[f64; NUM_INPUT + NUM_OUTPUT]; NUM_DATA_ITEMS],
    num_rows: usize,
    decimals: usize,
    indices: bool,
) {
    let len = matrix.len().to_string().len();
    matrix
        .iter()
        .enumerate()
        .take(num_rows)
        .for_each(|(i, item)| {
            if indices {
                print!("[{:>len$}]  ", i.to_string());
            }
            item.iter().for_each(|itm| {
                let v: f64 = *itm;
                if v >= 0.0 {
                    print!(" "); // '+'
                }
                print!("{:.decimals$}   ", v);
            });
            println!();
        });

    if num_rows < matrix.len() {
        println!(". . .");
        let last_row = matrix.len() - 1;
        if indices {
            print!("[{:>len$}]  ", last_row.to_string());
        }
        (0..matrix[last_row].len()).for_each(|j| {
            let v = matrix[last_row][j];
            if v >= 0.0 {
                print!(" "); // '+'
            }
            print!("{:.decimals$}   ", v);
        });
    }
    println!();
}

fn make_data() -> [[f64; NUM_INPUT + NUM_OUTPUT]; NUM_DATA_ITEMS] {
    let mut dn: DeepNet = DeepNet::new(); // make a DNN generator
    let mut rrnd = oorandom::Rand64::new(5); // to make random weights & biases, random input vals

    let wt_lo: f64 = -9.0;
    let wt_hi: f64 = 9.0;

    //let nw = dn.num_weights();
    let mut wts = [0.0; 303];

    wts.iter_mut().for_each(|item| {
        *item = (wt_hi - wt_lo) * rrnd.rand_float() + wt_lo;
    });
    dn.set_weights(&wts);

    let mut result = [[0.0; NUM_INPUT + NUM_OUTPUT]; NUM_DATA_ITEMS]; // make the result matrix holder

    let in_lo = -4.0; // pseudo-Gaussian scaling
    let in_hi = 4.0;

    result.iter_mut().for_each(|item| {
        let mut inputs = [0.0; NUM_INPUT]; // random input values

        inputs.iter_mut().for_each(|itm2| {
            *itm2 = (in_hi - in_lo) * rrnd.rand_float() + in_lo;
        });

        //ShowVector(inputs, 2);

        let probs = dn.compute_outputs(&inputs); // compute the outputs (as softmax probs) like [0.10, 0.15, 0.55, 0.20]

        //dn.Dump();
        //Console.ReadLine();
        //ShowVector(probs, 4);

        let outputs = probs_to_classes(&probs); // convert to outputs like [0, 0, 1, 0]

        let mut c = 0;
        inputs.iter().for_each(|itm2| {
            item[c] = *itm2;
            c += 1;
        });
        outputs.iter().for_each(|itm3| {
            item[c] = *itm3;
            c += 1;
        });
        //Console.WriteLine("");
    });

    result
}

fn probs_to_classes(probs: &[f64]) -> [f64; NUM_LAYERS] {
    let mut result = [0.0; NUM_LAYERS];
    let idx: usize = max_index(probs);
    result[idx] = 1.0;
    result
}

fn max_index(probs: &[f64]) -> usize {
    let mut max_idx = 0;
    let mut max_val = probs[0];

    probs.iter().enumerate().for_each(|(i, item)| {
        if *item > max_val {
            max_val = *item;
            max_idx = i;
        }
    });
    max_idx
}

pub struct DeepNet {
    pub rnd: oorandom::Rand64, // weight init and train shuffle
    pub n_hidden: [usize; 3],  // number hidden nodes, each layer

    pub i_nodes: [f64; NUM_INPUT],        // input nodes
    pub h_nodes: [[f64; 10]; NUM_OUTPUT], // array di array jagged
    pub o_nodes: [f64; NUM_OUTPUT],

    pub ih_weights: [[f64; 10]; NUM_INPUT], // input- 1st hidden
    pub hh_weights: [[[f64; 10]; 10]; 2], // hidden[2]]; hidden[1]]; hidden.len() - 1]// hidden-hidden
    pub ho_weights: [[f64; 10]; 10],      // hidden[hidden.len() - 1] // last hidden-output

    pub h_biases: [[f64; 10]; NUM_OUTPUT], // hidden node biases
    pub o_biases: [f64; NUM_OUTPUT],       // output node biases
    pub ih_gradient00: f64,                // one gradient to monitor
}

impl DeepNet {
    fn new() -> Self {
        let mut dn: DeepNet = DeepNet {
            ih_gradient00: 0.0,
            rnd: oorandom::Rand64::new(5),
            i_nodes: [0.0; NUM_INPUT],
            o_nodes: [0.0; NUM_OUTPUT],
            o_biases: [0.0; NUM_OUTPUT],
            h_nodes: [[0.0; 10]; NUM_OUTPUT],
            h_biases: [[0.0; 10]; NUM_OUTPUT],
            ih_weights: [[0.0; 10]; NUM_INPUT],
            ho_weights: [[0.0; 10]; 10],
            hh_weights: [[[0.0; 10]; 10]; 2],
            n_hidden: [10, 10, 10],
        };

        // make wts
        let lo = -0.10;
        let hi = 0.10;
        //let num_wts = dn.num_weights();
        let mut wts = [0.0; 303];

        wts.iter_mut().for_each(|item| {
            *item = (hi - lo) * dn.rnd.rand_float() + lo;
        });
        dn.set_weights(&wts);
        dn
    }

    fn num_weights(&self) -> usize {
        // total num weights and biases
        let ih_wts = NUM_INPUT * self.n_hidden[0];

        let mut hh_wts = 0;

        (0..self.n_hidden.len() - 1).for_each(|j| {
            hh_wts += self.n_hidden[j] * self.n_hidden[j + 1];
        });

        let ho_wts = self.n_hidden[self.n_hidden.len() - 1] * NUM_OUTPUT;

        let mut hbs = 0;

        (0..self.n_hidden.len()).for_each(|i| {
            hbs += self.n_hidden[i];
        });

        ih_wts + hh_wts + ho_wts + hbs + NUM_OUTPUT
    }

    /*
        fn get_weights(&mut self) -> Vec<f64> {
            // order: ihweights -> hhWeights[] -> hoWeights -> hBiases[] -> oBiases
            let nw = self.num_weights(NUM_INPUT, NUM_OUTPUT); // total num wts + biases
            let mut result = vec![0.0; nw];

            let mut ptr = 0; // pointer into result[]

            (0..NUM_INPUT).for_each(|i| {
                // input node
                (0..self.h_nodes[0].len()).for_each(|j| {
                    // 1st hidden layer nodes
                    result[ptr] = self.ih_weights[i][j];
                    ptr += 1;
                });
            });

            (0..NUM_LAYERS - 1).for_each(|h| {
                // not last h layer
                (0..self.n_hidden[h]).for_each(|j| {
                    // from node
                    (0..self.n_hidden[h + 1]).for_each(|jj| {
                        // to node
                        result[ptr] = self.hh_weights[h][j][jj];
                        ptr += 1;
                    });
                });
            });

            let hi = NUM_LAYERS - 1; // if 3 hidden layers (0,1,2) last is 3-1 = [2]
            (0..self.n_hidden[hi]).for_each(|j| {
                (0..NUM_OUTPUT).for_each(|k| {
                    result[ptr] = self.ho_weights[j][k];
                    ptr += 1;
                });
            });

            (0..NUM_LAYERS).for_each(|h| {
                // hidden node biases
                (0..self.n_hidden[h]).for_each(|j| {
                    result[ptr] = self.h_biases[h][j];
                    ptr += 1;
                });
            });
            (0..NUM_OUTPUT).for_each(|k| {
                result[ptr] = self.o_biases[k];
                ptr += 1;
            });

            result
        }

    */

    fn set_weights(&mut self, wts: &[f64]) {
        // order: ihweights - hhWeights[] - hoWeights - hBiases[] - oBiases
        let nw = self.num_weights(); // total num wts + biases
        if wts.len() != nw {
            panic!("Bad wts[] length in set_weights()")
        }
        let mut ptr = 0; // pointer into wts[]

        (0..NUM_INPUT).for_each(|i| {
            // input node
            (0..self.h_nodes[0].len()).for_each(|j| {
                self.ih_weights[i][j] = wts[ptr];
                ptr += 1;
            });
        });
        (0..NUM_LAYERS - 1).for_each(|h| {
            // not last h layers
            (0..self.n_hidden[h]).for_each(|j| {
                // from node
                (0..self.n_hidden[h + 1]).for_each(|jj| {
                    // to node
                    self.hh_weights[h][j][jj] = wts[ptr];
                    ptr += 1;
                });
            });
        });
        let hi = NUM_LAYERS - 1; // if 3 hidden layers (0,1,2) last is 3-1 = [2]
        (0..self.n_hidden[hi]).for_each(|j| {
            (0..NUM_OUTPUT).for_each(|k| {
                self.ho_weights[j][k] = wts[ptr];
                ptr += 1;
            });
        });

        (0..NUM_LAYERS).for_each(|h| {
            // hidden node biases
            (0..self.n_hidden[h]).for_each(|j| {
                self.h_biases[h][j] = wts[ptr];
                ptr += 1;
            });
        });
        (0..NUM_OUTPUT).for_each(|k| {
            self.o_biases[k] = wts[ptr];
            ptr += 1;
        });
    }

    fn compute_outputs(&mut self, x_values: &[f64]) -> [f64; NUM_LAYERS] {
        // 'xValues' might have class label or not
        // copy vals into iNodes

        self.i_nodes[..NUM_INPUT].copy_from_slice(&x_values[..NUM_INPUT]);

        // zero-out all hNodes, oNodes
        (0..NUM_LAYERS).for_each(|h| {
            (0..self.n_hidden[h]).for_each(|j| {
                self.h_nodes[h][j] = 0.0;
            });
        });

        (0..NUM_OUTPUT).for_each(|k| {
            self.o_nodes[k] = 0.0;
        });

        // input to 1st hid layer
        (0..self.n_hidden[0]).for_each(|j| {
            // each hidden node, 1st layer
            (0..NUM_INPUT).for_each(|i| {
                self.h_nodes[0][j] += self.ih_weights[i][j] * self.i_nodes[i];
            });
            // add the bias
            //hNodes[0][j] += hBiases[0][j];
            //// apply activation
            //hNodes[0][j] = Math.Tanh(hNodes[0][j]);

            //hNodes[0][j] = Math.Tanh(hNodes[0][j] + hBiases[0][j]);
            self.h_nodes[0][j] = tanh_levien(self.h_nodes[0][j] + self.h_biases[0][j]);
        });

        // each remaining hidden node
        (1..NUM_LAYERS).for_each(|h| {
            (0..self.n_hidden[h]).for_each(|j| {
                // 'to index'
                (0..self.n_hidden[h - 1]).for_each(|jj| {
                    // 'from index'
                    self.h_nodes[h][j] += self.hh_weights[h - 1][jj][j] * self.h_nodes[h - 1][jj];
                });
                //hNodes[h][j] += hBiases[h][j];  // add bias value
                //hNodes[h][j] =  Math.Tanh(hNodes[h][j]);  // apply activation

                //hNodes[h][j] = Math.Tanh(hNodes[h][j] + hBiases[h][j]);  // apply activation
                self.h_nodes[h][j] = tanh_levien(self.h_nodes[h][j] + self.h_biases[h][j]);
                // apply activation
            });
        });

        // compute ouput node values
        (0..NUM_OUTPUT).for_each(|k| {
            (0..self.n_hidden[NUM_LAYERS - 1]).for_each(|j| {
                self.o_nodes[k] += self.ho_weights[j][k] * self.h_nodes[NUM_LAYERS - 1][j];
            });
            self.o_nodes[k] += self.o_biases[k]; // add bias value
                                                 //Console.WriteLine("Pre-softmax output node [" + k + "] value = " + oNodes[k].ToString("F4"));
        });

        let ret_result: [f64; NUM_LAYERS] = DeepNet::soft_max(&self.o_nodes); // softmax activation all oNodes

        self.o_nodes[..NUM_OUTPUT].copy_from_slice(&ret_result[..NUM_OUTPUT]);
        ret_result
    } // compute_outputs

    fn soft_max(o_sums: &[f64]) -> [f64; NUM_LAYERS] {
        // does all output nodes at once so scale
        // doesn't have to be re-computed each time.
        // possible overflow . . . use max trick

        let mut sum = 0.0;
        for item in o_sums.iter() {
            sum += item.exp();
        }
        let mut result = [0.0; NUM_LAYERS];

        for i in 0..o_sums.len() {
            result[i] = o_sums[i].exp() / sum;
        }
        result // now scaled so that xi sum to 1.0
    }
    /*
        fn show_vector(vector: &[f64], dec: usize) {
            vector.iter().for_each(|i| {
                print!("{:.dec$}   ", i);
            });
            println!();
        }
    */
    fn accuracy(
        &mut self,
        data: &[[f64; NUM_INPUT + NUM_OUTPUT]; NUM_DATA_ITEMS],
        // verbose: bool,
    ) -> f64 {
        // percentage correct using winner-takes all
        let mut num_correct: f64 = 0.0;
        let mut num_wrong: f64 = 0.0;

        data.iter().for_each(|i| {
            let x_values = Vec::from_iter(i[0..NUM_INPUT].iter().cloned()); // get x-values
            let t_values = Vec::from_iter(i[NUM_INPUT..(NUM_INPUT + NUM_OUTPUT)].iter().cloned()); // get target values
            let y_values = self.compute_outputs(&x_values); // outputs using current weights
                                                            /*
                                                                        if verbose {
                                                                            DeepNet::show_vector(&y_values, 4);
                                                                            DeepNet::show_vector(&t_values, 4);
                                                                            println!();
                                                                            //Console.ReadLine();
                                                                        }
                                                            */
            let max_index0: usize = max_index(&y_values); // which cell in yValues has largest value?
            let t_max_index: usize = max_index(&t_values);

            if max_index0 == t_max_index {
                num_correct += 1.0;
            } else {
                num_wrong += 1.0;
            }
        });
        (num_correct * 1.0) / (num_correct + num_wrong)
    }

    fn error(
        &mut self,
        data: &[[f64; NUM_INPUT + NUM_OUTPUT]; NUM_DATA_ITEMS],
        // verbose: bool,
    ) -> f64 {
        // mean squared error using current weights & biases
        let mut sum_squared_error = 0.0;

        // walk thru each training case. looks like (6.9 3.2 5.7 2.3) (0 0 1)
        data.iter().for_each(|i| {
            let x_values = Vec::from_iter(i[0..NUM_INPUT].iter().cloned());
            let t_values = Vec::from_iter(i[NUM_INPUT..(NUM_INPUT + NUM_OUTPUT)].iter().cloned()); // get target values
            let y_values = self.compute_outputs(&x_values); // outputs using current weights
                                                            /*
                                                                        if verbose {
                                                                            DeepNet::show_vector(&y_values, 4);
                                                                            DeepNet::show_vector(&t_values, 4);
                                                                            println!();
                                                                            //Console.ReadLine();
                                                                        }
                                                            */
            (0..NUM_OUTPUT).for_each(|j| {
                let err = t_values[j] - y_values[j];
                sum_squared_error += err * err;
            });
        });

        sum_squared_error / ((data.len() * NUM_OUTPUT) as f64) // average per item
    } // error

    fn train(
        &mut self,
        train_data: &[[f64; NUM_INPUT + NUM_OUTPUT]; NUM_DATA_ITEMS],
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
        let mut ho_grads = [[0.0; NUM_OUTPUT]; 10]; // last_hidden layer - output weights grads
        let mut hh_grads = [[[0.0; 10]; 10]; 2];

        let mut ih_grads = [[0.0; 10]; NUM_INPUT]; // input-first_hidden wts gradients
                                                   // biases
        let mut ob_grads = [0.0; NUM_OUTPUT]; // output node bias grads
        let mut hb_grads = [[0.0; 10]; NUM_OUTPUT]; // hidden node bias grads

        // 2. each output node and each hidden node has a 'signal' == gradient without associated input (lower case delta in Wikipedia)
        let mut o_signals = [0.0; NUM_OUTPUT];
        let mut h_signals = [[0.0; 10]; NUM_OUTPUT];

        // 3. for momentum, each weight and bias needs to store the prev epoch delta
        // the structure for prev deltas is same as for Weights & Biases, which is same as for Grads

        let mut ho_prev_weights_delta = [[0.0; NUM_OUTPUT]; 10]; // last_hidden layer - output weights momentum term
        let mut hh_prev_weights_delta = [[[0.0; 10]; 10]; 2];

        let mut ih_prev_weights_delta = [[0.0; 10]; NUM_INPUT]; // input-first_hidden wts gradients
        let mut o_prev_biases_delta = [0.0; NUM_OUTPUT]; // output node bias prev deltas
        let mut h_prev_biases_delta = [[0.0; 10]; NUM_OUTPUT]; // hidden node bias prev deltas

        let mut epoch = 0;

        let mut sequence = [0; NUM_DATA_ITEMS];

        sequence.iter_mut().enumerate().for_each(|(i, item)| {
            *item = i;
        });

        let err_interval = max_epochs / show_every; // interval to check & display  Error

        while epoch < max_epochs {
            epoch += 1;
            if epoch % err_interval == 0 && epoch < max_epochs
            // display curr MSE
            {
                // using curr weights & biases
                print!(
                    "epoch = {:?}  MS error = {:.4} ",
                    epoch,
                    self.error(train_data)
                );
                println!("  accuracy = {:.4} ", self.accuracy(train_data));
                println!(
                    "input-to-hidden [0][0] gradient = {:.12}",
                    self.ih_gradient00
                );
                println!();
                //this.Dump();
                // Console.ReadLine();
            }

            DeepNet::shuffle(&mut sequence); // must visit each training data in random order in stochastic GD

            sequence.iter().for_each(|item| {
                // each train data item
                let idx = *item; // idx points to a data item

                let x_values = Vec::from_iter(train_data[idx][0..NUM_INPUT].iter().cloned()); // get x-values
                let t_values = Vec::from_iter(
                    train_data[idx][NUM_INPUT..(NUM_INPUT + NUM_OUTPUT)]
                        .iter()
                        .cloned(),
                ); // get target values

                self.compute_outputs(&x_values); // copy xValues in, compute outputs using curr weights & biases, ignore return

                // must compute signals from right-to-left
                // weights and bias gradients can be computed left-to-right
                // weights and bias gradients can be updated left-to-right

                // x. compute output node signals (assumes softmax) depends on target values to the right
                (0..NUM_OUTPUT).for_each(|k| {
                    let error_signal = t_values[k] - self.o_nodes[k]; // Wikipedia uses (o-t)
                    let derivative = (1.0 - self.o_nodes[k]) * self.o_nodes[k]; // for softmax (same as log-sigmoid) with MSE
                                                                                //derivative = 1.0;  // for softmax with cross-entropy
                    o_signals[k] = error_signal * derivative; // we'll use this for ho-gradient and hSignals
                });

                // x. compute signals for last hidden layer (depends on oNodes values to the right)
                let last_layer = NUM_LAYERS - 1;

                (0..self.n_hidden[last_layer]).for_each(|j| {
                    let mut sum: f64 = 0.0; // need sums of output signals times hidden-to-output weights
                    let derivative =
                        (1.0 + self.h_nodes[last_layer][j]) * (1.0 - self.h_nodes[last_layer][j]); // for tanh!
                    o_signals.iter().enumerate().for_each(|(k, item)| {
                        sum += item * self.ho_weights[j][k]; // represents error signal
                    });
                    h_signals[last_layer][j] = derivative * sum;
                });

                // x. compute signals for all the non-last layer hidden nodes (depends on layer to the right)
                (0..(NUM_LAYERS - 1)).rev().for_each(|h| {
                    // each hidden layer, right-to-left
                    (0..self.n_hidden[h]).for_each(|j| {
                        let mut sum = 0.0; // need sums of output signals times hidden-to-output weights
                                           // each node
                        let derivative = (1.0 + self.h_nodes[h][j]) * (1.0 - self.h_nodes[h][j]); // for tanh
                                                                                                  // derivative = hNodes[h][j];
                        (0..self.n_hidden[h + 1]).for_each(|jj| {
                            // layer to right of curr layer
                            sum += h_signals[h + 1][jj] * self.hh_weights[h][j][jj];
                        });
                        h_signals[h][j] = derivative * sum;
                    }); // j
                }); // h

                // at this point, all hidden and output node signals have been computed
                // calculate gradients left-to-right

                // x. compute input-to-hidden weights gradients using iNodes & hSignal[0]
                ih_grads.iter_mut().enumerate().for_each(|(i, item)| {
                    item.iter_mut().enumerate().for_each(|(j, itm1)| {
                        *itm1 = self.i_nodes[i] * h_signals[0][j]; // "from" input & "to" signal
                    });
                });

                // save the special monitored ihGradient00
                self.ih_gradient00 = ih_grads[0][0];

                // x. compute hidden-to-hidden gradients
                (0..NUM_LAYERS - 1).for_each(|h| {
                    (0..self.n_hidden[h]).for_each(|j| {
                        (0..self.n_hidden[h + 1]).for_each(|jj| {
                            hh_grads[h][j][jj] = self.h_nodes[h][j] * h_signals[h + 1][jj];
                        });
                    });
                });

                // x. compute hidden-to-output gradients
                ho_grads.iter_mut().enumerate().for_each(|(j, item)| {
                    o_signals.iter().enumerate().for_each(|(k, elem)| {
                        item[k] = self.h_nodes[last_layer][j] * elem;
                        // from last hidden, to oSignals
                    });
                });

                // compute bias gradients
                // a bias is like a weight on the left/before
                // so there's a dummy input of 1.0 and we use the signal of the 'current' layer

                // x. compute all hidden bias gradients
                // a gradient needs the "left/from" input and the "right/to" signal
                // for biases we use a dummy 1.0 input

                (0..NUM_LAYERS).for_each(|h| {
                    (0..self.n_hidden[h]).for_each(|j| {
                        hb_grads[h][j] = 1.0 * h_signals[h][j];
                    });
                });

                // x. output bias gradients
                (0..NUM_OUTPUT).for_each(|k| {
                    ob_grads[k] = 1.0 * o_signals[k];
                });

                // at this point all signals have been computed and all gradients have been computed
                // so can use gradients to update all weights and biases.
                // save each delta for the momentum

                // x. update input-to-first_hidden weights using ihWeights & ihGrads
                (0..NUM_INPUT).for_each(|i| {
                    (0..self.n_hidden[0]).for_each(|j| {
                        let delta = ih_grads[i][j] * learn_rate;
                        //ihWeights[i][j] += delta;
                        //ihWeights[i][j] += ihPrevWeightsDelta[i][j] * momentum;

                        self.ih_weights[i][j] += delta + ih_prev_weights_delta[i][j] * momentum;

                        ih_prev_weights_delta[i][j] = delta;
                    });
                });
                // other hidden-to-hidden weights using hhWeights & hhGrads
                (0..NUM_LAYERS - 1).for_each(|h| {
                    (0..self.n_hidden[h]).for_each(|j| {
                        (0..self.n_hidden[h + 1]).for_each(|jj| {
                            let delta = hh_grads[h][j][jj] * learn_rate;
                            //hhWeights[h][j][jj] += delta;
                            //hhWeights[h][j][jj] += hhPrevWeightsDelta[h][j][jj] * momentum;

                            self.hh_weights[h][j][jj] +=
                                delta + hh_prev_weights_delta[h][j][jj] * momentum;
                            hh_prev_weights_delta[h][j][jj] = delta;
                        });
                    });
                });

                // update hidden-to-output weights using hoWeights & hoGrads
                (0..self.n_hidden[last_layer]).for_each(|j| {
                    (0..NUM_OUTPUT).for_each(|k| {
                        let delta = ho_grads[j][k] * learn_rate;
                        //hoWeights[j][k] += delta;
                        //hoWeights[j][k] += hoPrevWeightsDelta[j][k] * momentum;

                        self.ho_weights[j][k] += delta + ho_prev_weights_delta[j][k] * momentum;
                        ho_prev_weights_delta[j][k] = delta;
                    });
                });

                // update hidden biases using hBiases & hbGrads
                (0..NUM_LAYERS).for_each(|h| {
                    (0..self.n_hidden[h]).for_each(|j| {
                        let delta = hb_grads[h][j] * learn_rate;
                        //hBiases[h][j] += delta;
                        //hBiases[h][j] += hPrevBiasesDelta[h][j] * momentum;

                        self.h_biases[h][j] += delta + h_prev_biases_delta[h][j] * momentum;
                        h_prev_biases_delta[h][j] = delta;
                    });
                });

                // update output biases using oBiases & obGrads
                (0..NUM_OUTPUT).for_each(|k| {
                    let delta = ob_grads[k] * learn_rate;
                    //oBiases[k] += delta;
                    //oBiases[k] += oPrevBiasesDelta[k] * momentum;

                    self.o_biases[k] += delta + o_prev_biases_delta[k] * momentum;
                    o_prev_biases_delta[k] = delta;
                });

                // Whew!
            }); // for each train data item
        } // while

        //let _best_wts = self.get_weights();
    } // Train

    fn shuffle(sequence: &mut [usize]) // instance method
    {
        let mut rnd = oorandom::Rand32::new(5);
        (0..sequence.len()).for_each(|i| {
            let r = rnd.rand_range(std::ops::Range {
                start: i as u32,
                end: sequence.len() as u32,
            });
            sequence.swap(r as usize, i);
        });
    } // Shuffle
}
