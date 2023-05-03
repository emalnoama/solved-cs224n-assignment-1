Download Link: https://assignmentchef.com/product/solved-cs224n-assignment-1
<br>
<strong>CS 224N: Assignment #1</strong>




<h1>1           Softmax</h1>

<ul>

 <li>(5 points) Prove that softmax is invariant to constant offsets in the input, that is, for any input vector <em>x </em>and any constant <em>c</em>, softmax(<em>x</em>) = softmax(<em>x </em>+ <em>c</em>)</li>

</ul>

where <em>x </em>+ <em>c </em>means adding the constant <em>c </em>to every dimension of <em>x</em>. Remember that

(1)

<em>Note: In practice, we make use of this property and choose c </em>= −max<em><sub>i </sub>x<sub>i </sub>when computing softmax probabilities for numerical stability (i.e., subtracting its maximum element from all elements of </em><em>x).</em>

<ul>

 <li>(5 points) Given an input matrix of N rows and D columns, compute the softmax prediction for each row using the optimization in part (a). Write your implementation in py. You may test by executing python q1softmax.py.</li>

</ul>

<em>Note: The provided tests are not exhaustive. Later parts of the assignment will reference this code so it is important to have a correct implementation. Your implementation should also be efficient and vectorized whenever possible (i.e., use numpy matrix operations rather than for loops). A non-vectorized implementation will not receive full credit!</em>

<h1>2           Neural Network Basics</h1>

<ul>

 <li>(3 points) Derive the gradients of the sigmoid function and show that it can be rewritten as a function of the function value (i.e., in some expression where only <em>σ</em>(<em>x</em>), but not <em>x</em>, is present). Assume that the input <em>x </em>is a scalar for this question. Recall, the sigmoid function is</li>

 <li>(3 points) Derive the gradient with regard to the inputs of a softmax function when cross entropy loss is used for evaluation, i.e., find the gradients with respect to the softmax input vector <em>θ</em>, when the prediction is made by <em>y</em>ˆ = softmax(<em>θ</em>). Remember the cross entropy function is</li>

</ul>

<em>CE</em>(<em>y,</em><em>y</em>ˆ) = −<sup>X</sup><em>y<sub>i </sub></em>log(ˆ<em>y<sub>i</sub></em>)                                                                          (3)

<em>i</em>

where <em>y </em>is the one-hot label vector, and <em>y</em>ˆ is the predicted probability vector for all classes. (<em>Hint: you might want to consider the fact many elements of </em><em>y are zeros, and assume that only the k-th dimension of </em><em>y is one.</em>)

<ul>

 <li>(6 points) Derive the gradients with respect to the <em>inputs </em><em>x </em>to an one-hidden-layer neural network (that is, find <em><sub>∂</sub><u><sup>∂J</sup></u></em><em><sub>x </sub></em>where <em>J </em>= <em>CE</em>(<em>y,</em><em>y</em>ˆ) is the cost function for the neural network). The neural network employs sigmoid activation function for the hidden layer, and softmax for the output layer. Assume the one-hot label vector is <em>y</em>, and cross entropy cost is used. (Feel free to use <em>σ</em><sup>0</sup>(<em>x</em>) as the shorthand for sigmoid gradient, and feel free to define any variables whenever you see fit.)</li>

</ul>

Recall that the forward propagation is as follows

<em>h </em>= sigmoid(<em>xW</em><sub>1 </sub>+ <em>b</em><sub>1</sub>)                                         <em>y</em>ˆ = softmax(<em>hW</em><sub>2 </sub>+ <em>b</em><sub>2</sub>)

Note that here we’re assuming that the input vector (thus the hidden variables and output probabilities) is a row vector to be consistent with the programming assignment. When we apply the sigmoid function to a vector, we are applying it to each of the elements of that vector. <em>W<sub>i </sub></em>and <em>b<sub>i </sub></em>(<em>i </em>= 1<em>,</em>2) are the weights and biases, respectively, of the two layers.

<ul>

 <li>(2 points) How many parameters are there in this neural network, assuming the input is <em>D<sub>x</sub></em>-dimensional, the output is <em>D<sub>y</sub></em>-dimensional, and there are <em>H </em>hidden units?</li>

 <li>(4 points) Fill in the implementation for the sigmoid activation function and its gradient in py. Test your implementation using python q2sigmoid.py. <em>Again, thoroughly test your code as the provided tests may not be exhaustive.</em></li>

 <li>(4 points) To make debugging easier, we will now implement a gradient checker. Fill in the implementation for gradchecknaive in py. Test your code using python q2gradcheck.py.</li>

 <li>(8 points) Now, implement the forward and backward passes for a neural network with one sigmoid hidden layer. Fill in your implementation in py. Sanity check your implementation with python q2neural.py.</li>

</ul>

<h1>3           word2vec</h1>

<ul>

 <li>(3 points) Assume you are given a predicted word vector <em>v<sub>c </sub></em>corresponding to the center word <em>c </em>for skipgram, and word prediction is made with the softmax function found in word2vec models</li>

</ul>

<em>y</em>ˆ<em><sub>o </sub></em>= <em>p</em>(<em>o</em>                                                               (4)

where <em>w </em>denotes the w-th word and <em>u<sub>w </sub></em>(<em>w </em>= 1<em>,…,W</em>) are the “output” word vectors for all words in the vocabulary. Assume cross entropy cost is applied to this prediction and word <em>o </em>is the expected word (the <em>o</em>-th element of the one-hot label vector is one), derive the gradients with respect to <em>v<sub>c</sub></em>.

<em>Hint: It will be helpful to use notation from question 2. For instance, letting </em><em>y</em>ˆ <em>be the vector of softmax predictions for every word, </em><em>y as the expected word vector, and the loss function</em>

<em>J<sub>softmax</sub></em>−<em><sub>CE</sub></em>(<em>o,</em><em>v<sub>c</sub>,</em><em>U</em>) = <em>CE</em>(<em>y,</em><em>y</em>ˆ)                                                                      (5)

where <em>U </em>= [<em>u</em><sub>1</sub><em>,</em><em>u</em><sub>2</sub><em>,</em>··· <em>,</em><em>u<sub>W</sub></em>] is the matrix of all the output vectors. <em>Make sure you state the orientation of your vectors and matrices.</em>

<ul>

 <li>(3 points) As in the previous part, derive gradients for the “output” word vectors <em>u<sub>w</sub></em>’s (including <em>u<sub>o</sub></em>).</li>

 <li>(6 points) Repeat part (a) and (b) assuming we are using the negative sampling loss for the predicted vector <em>v<sub>c</sub></em>, and the expected output word is <em>o</em>. Assume that <em>K </em>negative samples (words) are drawn, and they are <strong>1</strong><em>,</em>·· <em>,</em><em>K</em>, respectively for simplicity of notation (<em>o /</em>∈ {1<em>,…,K</em>}). Again, for a given word, <em>o</em>, denote its output vector as <em>u<sub>o</sub></em>. The negative sampling loss function in this case is</li>

</ul>

))                                    (6)

where <em>σ</em>(·) is the sigmoid function.

After you’ve done this, describe with one sentence why this cost function is much more efficient to compute than the softmax-CE loss (you could provide a speed-up ratio, i.e., the runtime of the softmaxCE loss divided by the runtime of the negative sampling loss).

<em>Note: the cost function here is the negative of what Mikolov et al had in their original paper, because we are doing a minimization instead of maximization in our code.</em>

<ul>

 <li>(8 points) Derive gradients for all of the word vectors for skip-gram and CBOW given the previous parts and given a set of context words [word<em><sub>c</sub></em><sub>−<em>m</em></sub><em>,…,</em>word<em><sub>c</sub></em><sub>−1</sub><em>,</em>word<em><sub>c</sub>,</em>word<em><sub>c</sub></em><sub>+1</sub><em>,…,</em>word<em><sub>c</sub></em><sub>+<em>m</em></sub>], where <em>m </em>is the context size. Denote the “input” and “output” word vectors for word<em><sub>k </sub></em>as <em>v<sub>k </sub></em>and <em>u<sub>k </sub></em></li>

</ul>

<em>Hint: feel free to use F</em>(<em>o,</em><em>v<sub>c</sub></em>) <em>(where </em><em>o is the expected word) as a placeholder for the J<sub>softmax</sub></em><sub>−<em>CE</em></sub>(<em>o,</em><em>v<sub>c</sub>,…</em>) <em>or J<sub>neg</sub></em><sub>−<em>sample</em></sub>(<em>o,</em><em>v<sub>c</sub>,…</em>) <em>cost functions in this part — you’ll see that this is a useful abstraction for the coding part. That is, your solution may contain terms of the form </em>.

Recall that for skip-gram, the cost for a context centered around <em>c </em>is

<em>J</em>skip-gram(word<em>c</em>−<em>m…c</em>+<em>m</em>) = X <em>F</em>(<em>w</em><em>c</em>+<em>j</em><em>,</em><em>v</em><em>c</em>)                                                             (7)

−<em>m</em>≤<em>j</em>≤<em>m,j</em>6=0

where <em>w<sub>c</sub></em><sub>+<em>j </em></sub>refers to the word at the <em>j</em>-th index from the center.

CBOW is slightly different. Instead of using <em>v<sub>c </sub></em>as the predicted vector, we use <em>v</em>ˆ defined below. For (a simpler variant of) CBOW, we sum up the input word vectors in the context

<em>v</em>ˆ = X <em>v</em><em>c</em>+<em>j                                                                                                                                 </em>(8)

−<em>m</em>≤<em>j</em>≤<em>m,j</em>6=0

then the CBOW cost is

<em>J</em>CBOW(word<em>c</em>−<em>m…c</em>+<em>m</em>) = <em>F</em>(<em>w</em><em>c,</em><em>v</em>ˆ)                                                                   (9)

<em>Note: To be consistent with the </em><em>v</em>ˆ <em>notation such as for the code portion, for skip-gram </em><em>v</em>ˆ = <em>v<sub>c</sub>.</em>

<ul>

 <li>(12 points) In this part you will implement the word2vec models and train your own word vectors with stochastic gradient descent (SGD). First, write a helper function to normalize rows of a matrix in py. In the same file, fill in the implementation for the softmax and negative sampling cost and gradient functions. Then, fill in the implementation of the cost and gradient functions for the skipgram model. When you are done, test your implementation by running python q3word2vec.py. <em>Note: If you choose not to implement CBOW (part h), simply remove the NotImplementedError so that your tests will complete.</em></li>

 <li>(4 points) Complete the implementation for your SGD optimizer in py. Test your implementation by running python q3sgd.py.</li>

 <li>(4 points) Show time! Now we are going to load some real data and train word vectors with everything you just implemented! We are going to use the Stanford Sentiment Treebank (SST) dataset to train word vectors, and later apply them to a simple sentiment analysis task. You will need to fetch the datasets first. To do this, run sh getdatasets.sh. There is no additional code to write for this part; just run python q3py.</li>

</ul>

<em>Note: The training process may take a long time depending on the efficiency of your implementation </em><em>(an efficient implementation takes approximately an hour). Plan accordingly!</em>

When the script finishes, a visualization for your word vectors will appear. It will also be saved as q3wordvectors.png in your project directory. <strong>Include the plot in your homework write up. </strong>Briefly explain in at most three sentences what you see in the plot.

<ul>

 <li>(Extra credit: 2 points) Implement the CBOW model in py. <em>Note: This part is optional but the gradient derivations for CBOW in part (d) are not!</em>.</li>

</ul>

<h1>4           Sentiment Analysis</h1>

Now, with the word vectors you trained, we are going to perform a simple sentiment analysis. For each sentence in the Stanford Sentiment Treebank dataset, we are going to use the average of all the word vectors in that sentence as its feature, and try to predict the sentiment level of the said sentence. The sentiment level of the phrases are represented as real values in the original dataset, here we’ll just use five classes:

“very negative (−−)”, “negative (−)”, “neutral”, “positive (+)”, “very positive (++)”

which are represented by 0 to 4 in the code, respectively. For this part, you will learn to train a softmax classifier, and perform train/dev validation to improve generalization.

<ul>

 <li>(2 points) Implement a sentence featurizer. A simple way of representing a sentence is taking the average of the vectors of the words in the sentence. Fill in the implementation in py.</li>

 <li>(1 points) Explain in at most two sentences why we want to introduce regularization when doing classification (in fact, most machine learning tasks).</li>

 <li>(2 points) Fill in the hyperparameter selection code in py to search for the “optimal” regularization parameter. Attach your code for chooseBestModel to your written write-up. You should be able to attain at least 36.5% accuracy on the dev and test sets using the pretrained vectors in part (d).</li>

 <li>(3 points) Run python q4sentiment.py –yourvectors to train a model using your word vectors from q3. Now, run python q4sentiment.py –pretrained to train a model using pretrained GloVe vectors (on Wikipedia data). Compare and report the best train, dev, and test accuracies. Why do you think the pretrained vectors did better? Be specific and justify with 3 distinct reasons.</li>

 <li>(4 points) Plot the classification accuracy on the train and dev set with respect to the regularization value for the pretrained GloVe vectors, using a logarithmic scale on the x-axis. This should have been done automatically. <strong>Include </strong><strong>png in your homework write up. </strong>Briefly explain in at most three sentences what you see in the plot.</li>

 <li>(4 points) We will now analyze errors that the model makes (with pretrained GloVe vectors). When you ran python q4sentiment.py –pretrained, two files should have been generated. Take a look at png and <strong>include it in your homework writeup</strong>. Interpret the confusion matrix in at most three sentences.</li>

 <li>(4 points) Next, take a look at txt. Choose 3 examples where your classifier made errors and briefly explain the error and what features the classifier would need to classify the example correctly (1 sentence per example). Try to pick examples with different reasons.</li>

</ul>