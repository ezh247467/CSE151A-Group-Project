# CSE151A-Group-Project
Ethan Huang \
Noah Danan \
[Juypter Notebook](Group_Project.ipynb)
## Introduction
Chess is a popular game that uses strategic thinking and tactical prowess, played for centuries and still counting. This dataset that we will be using comprises approximately 16 million unique chess positions, each evaluated by the Stockfish chess engine at a depth of 22. Stockfish, a state-of-the-art chess analysis tool, provides precise and detailed evaluations of positions, making this dataset highly valuable for research in artificial intelligence, game theory, and machine learning. The extensive depth of 22 ensures a deep and thorough analysis of each position, offering insights into optimal moves and strategies. This dataset can be instrumental in training advanced machine learning models, developing new chess algorithms, and conducting comprehensive studies on chess strategy and position evaluation.
## Data Exploration
### Since our dataset's only independent variable does not work well as either a continuous value or categorical variable, we will be extracting information from it.

The two main features we will be extracting from the chess position are **material advantage** (continuous value) and **development of the pieces** (can be scored into a continuous value). These two attributes will be the data we will use for our modeling as the independent variable to predict the Evaluation Score that has been provided by Stockfish.

---
Material Advantage is in the range of [-39, 39] and can be normalized into a smaller scale, such [-1, 1] or [0, 1]. 
\
Development of pieces will be scored using our own evaluation function (the criteria for scoring can be read below) and weighted on a 0-100 scale (where positive is white and negative is black like material advantage). This also can be normalized to a [-1,1] scale.
\
The Evaluation score by Stockfish is a bit tricky. Due to a depth of 22, we may see abnormally large evaluation scores (greater than 300) which would greatly affect the scale and putting the distribution at a very difficult to predict area. In addition, the Evaluation data includes forced mate notation, which provides a number moves necessary to achieve checkmate rather than a number to evaluate the position.

## Data Preprocessing
For our 'material advantage' and 'development of pieces' features, we will normalize our values to [-1, 1] for both attributes as it should save on computational resources since we will be computing with smaller numbers while preserving the scale. As for the Evaluation values, due to the incredible discrepancy and gaps in values, we believe that limiting the values to $\pm300$ and having any forced checkmates be equal to those max/min values. Given that an evaluation of (the absolute value of) anything greater than 39 would imply a similar advantage of the maximum material advantage, we believe there would be no significant impact if we were to equate any forced checkmate evaluation to be the same as a 300 point evaluation.

## Development of Pieces
#### The following function will check how well "developed" a player's core pieces are. There are many factors<sup>[^1]</sup> to this attribute, so our evaluation would certainly not be the most accurate. We also have our own arbitrary weights for evaluating as we cannot be completely sure how much "better developed" a piece is in relation to other types of pieces. The criteria that we will keep in mind for our evaluation function are the following.
- Queen
  - Penalty for early game development (first 5 moves) (0% - 5%)
  - **Queen mobility** (15%, 20% if midgame)
- Pawn
  - ~~Pawn structure~~ (Difficult and computationally expensive to evaluate)
  - **Pawn Center** (d4 or e4 defended by pawns or creating a double pawn on either d or e columns) (10%)
  - Penalty for "d" and "e" pawns being blocked at their starting squares (10%)
  - Late game (25+ moves): Penalty for pawns that are still near start position (0% - 5%)
- Knight
  - Less value if there are less pawns (5%)
  - **Knight mobility** (15%)
  - Penalty if undefended (10%)
- Bishop
  - **Bishop mobility** (greater emphasis on forward mobility) (15%)
  - Bishop pair is considered marginally stronger than Bishop Knight and Knight Knight (5%)
  - ~~Color Weakness (missing a bishop and poor pawn structure)~~ (See Pawn)
  - Penalty if undefended (10%)

> **Mobility omits squares controlled by enemy pawns**


[^1]: https://www.chessprogramming.org/Evaluation_of_Pieces