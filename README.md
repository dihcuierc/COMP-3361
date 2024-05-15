# COMP 3361 Natural Langauge Processing Assignment 2

Understanding of Transformers

## Part 1:

- Learning to code a Transformer Encoder.
- 3-class classification task. Given a string of characters, the task is to predict, for each position in the string, how many times the character at that position occurred before, maxing out at 2.

![output image](a2p12/part1output.png)

Output on test words:

- Accuracy: 19870 / 20000 = 0.993500

## Part 2:

- Learning to handle continuous stream of data that exceeds transformer input length.
- Perform chunking to split long streams of characters into a few chunks to feed to transformer model.
- Outputting the log probabilites of the next character given an input string of characters. (27 class classification: alphebet + 'space' character)

![output image](a2p12/part2output.png)
Output on test words:

- log_prob: -970.0538951586932
- avg_log_prob: -1.9401077903173864,
- perplexity: 6.959501097040466
