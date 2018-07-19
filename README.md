<h1>Credits to Michael Nielsen</h1>
<p>I do own this code but not the ideas behind of it. All of them come from Michael Nielsen's free book <a href="http://neuralnetworksanddeeplearning.com/index.html"><q>Neural Networks and Deep Learning</q></a>.</p>
<ul>
	<li>
		<h1>
			<a href="./cheese/cheese.py">
				Cheese.py
			</a>
		</h1>
	</li>
	<p>I made this code in order to test the <b>Perceptron</b> concept. The exposition in the original source:</p>
	<p><i><q> Suppose the weekend is coming up, and you've heard that there's going to be a cheese festival in your city. You like cheese, and are trying to decide whether or not to go to the festival. You might make your decision by weighing up three factors:</q>
	<ol>
		<li>Is the weather good?</li>
		<li>Does your boyfriend or girlfriend want to accompany you?</li>
		<li>Is the festival near public transit? (You don't own a car).</li>
	</ol>
	<q>We can represent these three factors by corresponding binary variables <b>x<sub>1</sub></b>, <b>x<sub>2</sub></b> and <b>x<sub>3</sub></b></q>
	<br>
	[...]
	<br>
	<q>
	By varying the weights and the threshold, we can get different models of decision-making.
	</q></i></p>
	<p>So, I was moved to try and test it on code. This was the input and output I used to train my neural network:</p>
	<table align="center">
		<tr>
			<td><b>x<sub>1</sub></b></td>
			<td><b>x<sub>2</sub></b></td>
			<td><b>x<sub>3</sub></b></td>
			<td><b>Ouput</b></td>
		</tr>
		<tr>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
		</tr>
		<tr>
			<td>1</td>
			<td>1</td>
			<td>1</td>
			<td>1</td>
		</tr>
		<tr>
			<td>1</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
		</tr>
		<tr>
			<td>1</td>
			<td>1</td>
			<td>0</td>
			<td>1</td>
		</tr>
	</table>
	<p>Then I used the random library for python to assign random weights to each input. After that, I iterated 10000 times using the sigmoid function and voilà. At the end of this process, the network was ready to take decisions by its own from different inputs.</p>
</ul>