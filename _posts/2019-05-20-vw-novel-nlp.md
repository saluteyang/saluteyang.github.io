---
layout: post
title: Can NLP Help Us Understand Modernist Texts?
---

*To the Lighthouse* is the middle of Virginia Woolf's five major novels and considered by many to be her finest. She had written *Mrs Dalloway* a few years earlier, so one might say that she had already broken substantial ground on her project of the modernist novel. And in this blogger's humble opinion, *To the Lighthouse* is less overtly concerned about form and experimentation than some of her other novels. It seems suffused with feeling and a genuine reflection of the central relationships of man and wife, and of children and parents.

On learning more about the novel, it is perhaps revealing that large chunks of the text were autobiographical. In Woolf's own words, the plan of the novel was to begin in Eden, with "father, mother & child in the garden" but then interruption would ensure in the form of mother's death with an ending of "the sail to the lighthouse." Much of the trajectory would mirror the true events in Woolf's own life: her family having spent summers in Talland House near St. Ives, looking across from the Godrevy Lighthouse; her mother's sudden death when Virginia was still a budding teanager; the grown-up children's revisit to the old vacation grounds nearby after years being away...

How much, if at all, can **Natural Language Processing** add to our understanding of an influential modernist text like *To the Lighthouse*? Would topic modeling be able to discern the central themes of the novel? Would sentiment analysis chart the emotional ebb and flow of the loosely constructed storyline? I began this project with some reservations.

The result turned out to be quite surprising. Depending on how we partitioned the text (and adding stop words judiciously before vectorizing), the topics that emerged would anchor around the male characters and female characters in one iteration of the model and, in another, closely hew to the subthemes of filial affection, artistic pursuit and the impressionistic brushstrokes of time and place.

#### Topics when dividing the novel into five equal parts, using TF-IDF and non-negative matrix factorization
![Image topics_by_partition]({{ site.url }}/images/vw_ttl_topics_partition.png)

Having excluded the proper nouns including most names from the vectorizer output, we see a clear clustering of words either describing the male characters or the events surrounding them (particularly recognizable in this word cloud are Mr. Ramsey, Charles Tansley), and of words emanating from the female characters (predominantly, Mrs. Ramsey). The centrality of husband and wife is aparent here among the topics, just as it is in the novel. As Woolf wrote in her diary of her own near-obsession with her parents:

> I used to think of father and mother daily; but writing it, laid them in my mind... I was obsessed by them both, unhealthily; & writing of them was a necessary act.

#### Topics when dividing the novel using the author's chapters

When I chunked up the novel more finely by chapter, a different set of topics now came to the fore. Some of the topics are quite subtle and require a degree of familiary with the subtler aspects of the novel. But subsuming the matrix output (lists of words in each topic) under the following headings would only require a small bit of associative imagination. How the mechanistic and bloodless matrix operation of factorization was able to generate such results was perhaps the more surprising bit.

![Image topics_by_chapter]({{ site.url }}/images/vw_ttl_topics_chapters.png)

Less of a science than topic modeling is the machine learning technique of using pretrained models of word polarities to gather information about the sentiment of a document. Here again, the way I divvied up the text would greatly affect whether I got an intelligible emotional cartography.

One chart generated by a single listcomp 
```python
[TextBlob(x).sentiment for x in docs]
```
involving the <code>TextBlob</code> package would convincingly identify the point near the nadir of the novel's narrative, where Mrs. Ramsey passed away in the section *Time Passes* in literal parantheses. One would be right to suspect that the characters reflecting on the emptiness left by the disappearance of the ballast of the Ramsey family was probably what the textual analysis here picked up on.

![Image sentiments_10_part]({{ site.url }}/images/vw_ttl_textblob_sentiment.png)

### Final thoughts
Of course, it would be quite soothing to chalk the topic and sentiment results up to some innate understanding of this complicated text by our forthcoming robot overlord. But just as all the great machinery of deep learning is, to simply to the extreme, some mapping (and repoduction) of patterns in data, the same can be said of the illusory way that the machine seemed here to lumber close to how we humans approach a text.

It would be quite tragic indeed if anyone, after reading *To the Lighthouse* and experiencing its many literary rewards, only takes away from the text some (spline-interpolated) charting of sentiments.