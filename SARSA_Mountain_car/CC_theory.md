# Coarse coding theory

In order to represent a continuous state space. You cover the entire state with overlapping shapes (circles, tiles etc) called "features". Then, when mapping a state coarsly, you simply check what features are "active" (your continuous are somewhere inside the shape) and which features are not. The active ones get a 1 value and the inactive ones get 0. This is called binary coarse coding (0 & 1), but you could also do it in a fuzzy way: "how much of this feature is present".

## Tile coding
Is a form of coarse coding we are going to use in this project. It is used for multidimentional continuous state spaces, and might be the best form of coarse coding for computational efficiency. In tile coding you use square tiles covering the entire state space. One layer of square tiles is called a "tiling". If you only used one single such tiling then it would not be coarse coding, as you could only have one active feature at the same time => it would just be a less accurate representation of the state. You need multiple, overlapping tilings to get a coarse code, and that is alo **required for this project.**

**Example**: a 2D state, x and y coordinates.
- Make 4 tilings (4 layers of tiles) covering the actual x-y state space. Each tiling is made out of 4x4 tiles/features. 
- The feature vector X(s) = a 4x4x4 = 16 sized array of all the features. State is the input. 
- The 4 tilings do not perfectly overlap but are shifted. If a state is represented as a x,y-coordinate, than exactly one feature of each layer will be 1, the rest 0.
- The feature vector is then the input feature used to train the NN.

