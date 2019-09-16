# Fake News Detection: Towards an Interpretable Approach

Fake news poses a significant threat to undermining the democratic process. Thankfully, much machine learning and natural language processing research has focused on fake news detection. However, these approaches nearly always use complex deep neural networks that lack interpretability, and thus do not allow us to reach new conclusions about the nature of fake news. In order to close this gap, this research adopts an **interpretable approach**, with the overall objective of producing a highly accurate fake news classification model, without compromising on interpretability.

This research approaches the problem of fake news classification entirely from a **text-based perspective**. No metadata such as article author or source is used to predict veracity. Three approaches were taken for feature engineering: **document vectors** using word embeddings, text frequency using **document-term matrices**, and hand crafted descriptive **text features**. The third approach received the most focus, as it retains the most interpretability and maintains a reasonably low number of predictors.

Text-based features were built using a series of tools and libraries, including the R package *tidytext*, the word processing engine *LIWC*, and the *Stanford CoreNLP* language tools. These features ranged from simple statistics such as mean sentence word count to syntactic variables created using CoreNLP's constituency parsers to psychological measures from LIWC. The code for creating these features can be seen in [text_features.R](https://github.com/CaioBrighenti/fake-news/blob/master/text_features.R) and the final outputs can be found in the [/annotations](https://github.com/CaioBrighenti/fake-news/tree/master/features) and [/features](https://github.com/CaioBrighenti/fake-news/tree/master/features) folders.

This is an ongoing project, and is currently in the model fitting and predictor importance analysis phase. Preliminary results using a simple logistic regression classifier have achieved approximately sensitivity, specificity and accuracy values ranging from 0.7-0.8 on a holdout test set. The objective of this project is to reach novel conclusions regarding the textual nature of fake news. To this end, the current focus is on exploring which variables are contributing the most to predictions, and what we might learn from this.

## Research summary poster
*Note: this poster was created approximately a month into the research, and does not represent the current state of the work.*

![Reseach Poster](https://github.com/CaioBrighenti/fake-news/blob/master/plots/Poster.png?raw=true)

## File Structure


## Datasets and main libraries used
*Note: this is not a comprehensive list of all packages used.*

* [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) - Dataset
* [LIAR](https://arxiv.org/abs/1705.00648) - Dataset
* [tidyverse](https://www.tidyverse.org/) - Collection of R packages for data science
* [tidytext](https://cran.r-project.org/web/packages/tidytext/index.html) - R package for tidyverse-style text analysis and NLP
* [LIWC2015](http://liwc.wpengine.com/) - Computerized text analysis tool\
* [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) - Text annotation toolkit
* [text2vec](http://text2vec.org/) - R package for text analysis and NLP

## Author

* **Caio Brighenti** - [CaioBrighenti](https://github.com/CaioBrighenti)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Thank you to the [*Colgate University Divison of Natural Sciences and Mathematics*](https://www.colgate.edu/academics/departments-programs/division-natural-sciences-and-mathematics) for funding this research.
* Thank you to Professor Will Cipolli at Colgate University for providing invaluable mentorship and support
