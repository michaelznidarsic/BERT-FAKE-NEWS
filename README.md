# BERT-FAKE-NEWS
Predicts news text's reliability with 91%+ validation accuracy. Uses Google BERT encoding as input for a Deep Bidirectional-LSTM Neural Network. Dataset consists of decent-length articles balanced for political leaning and spanning a diverse spectrum of reliability to fit the real-world newsscape. Initial research for this model available at https://github.com/michaelznidarsic/FakeNewsDetection

This model uses the BERT-Mini 256D embedding layer made available by Google at https://github.com/google-research/bert. The layer was initialized with the pre-trained weights, but fine-tuned during model training.

Articles from the AllTheNews corpus and Several27's FakeNewsCorpus were selected to create a new dataset. Publications to be labelled "Reliable" were handpicked to comprise a group of fact-based articles with reasonable levels of bias. To remove the possibility of the model conflating a given political leaning with reliability, equal numbers of articles leaning Left and Right, with many also included from the Center. The political leaning and reliability of the publications included were partly advised by mediabiasfactcheck.com.

To reflect the broad continuum of reliability and bias in the real world, the articles for the "Unreliable" label were drawn equally from the "Fake", "Bias", and "Political" labels in FakeNewsCorpus. This was done to introduce the concept of a "gray area" to the model, so that is would not be trained or evaluated only on the difference between professional journalism and informal blogs or social media. 

Phrases that would allow the model to "cheat" (e.g. "New York Times" or "Breitbart") were removed from training and evaluation data. They would allow the model to pick up on which publisher had written an article and equate that to reliability. This model was built to be deployed on any news texts of significant length, especially those for which an author or published is not known to the reader. As such, it should not rely on finding the article's source. If it were available, a reader could do that much more effectively.





