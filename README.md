# SongBird
## ACM RecSys Challenge
This year's ACM(Association for Computing Machinery) RecSys(Recommender System) challenge is sponsored by Spotify. The challenge focuses on music recommendation, specifically the challenge of automated playlist continuation. By suggesting appropriate songs to add to a playlist, a recommender system can greatly augment user engagement by making playlist creation easier, as well as extending listening beyond the end of existing playlists.

## Challenge Goal
The goal of the challenge is to develop a system for the task of automatic playlist continuation. Given a list of playlist metadata, the system should be able to recommend a set of tracks that can be added to it, thereby ‘continuing’ the playlist.

## Prior Work
Prior work in the field of music recommendation has primarily focused on shallow unsupervised approaches such as collaborative filtering and clustering, along with deep supervised approaches such as spectrogram analysis through CNNs. However, these shallow approaches suffer from a lack of representational ability, and the deep approaches suffer from the need to have access to large amounts of dense raw audio data, which is often times cumbersome to pre-process and train.

## Our Approach
We now present a novel unsupervised hybrid framework that leverages a Long Short Term Memory model (LSTM), Skip-Gram model, and Siamese Network. Specifically, we use a Siamese Network to learn an embedding of a metadata feature vector in a higher-dimensional space, where the distance between two songs mirrors a novel pairwise IOU metric. This captures the similarity between songs. We also use a Skip-Gram model to learn an embedding of songs in the playlist space. Finally we train an LSTM model on a combination of these embeddings to predict(or recommend) the next song(s) given a playlist of songs.

We train our model on a processed version of the Spotify Million Playlist Dataset, consisting of a million playlists of anywhere between 5 and 250 individual tracks each, and corresponding metadata such as tempo, key, valence, etc. Our model is able to learn complex representations between songs in a relatively small, sparse feature space.
