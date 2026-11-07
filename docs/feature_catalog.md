# Feature catalog

Generated from `src/features/registry.py`. Features are computed from the train split only, as of the split cutoff, and read through `FeatureStore` for both training and prediction.

| feature | entity | type | table | description | formula |
|---|---|---|---|---|---|
| `user_activity_count` | user | int | user_features | Number of ratings the user gave before the cutoff | `count(train rows of user)` |
| `user_avg_rating` | user | float | user_features | Mean rating given by the user | `mean(rating)` |
| `user_rating_std` | user | float | user_features | Spread of the user's ratings, 0 for a single rating | `std(rating)` |
| `user_positive_ratio` | user | float | user_features | Share of the user's ratings that count as liked | `mean(rating >= relevance_threshold)` |
| `user_days_since_last_interaction` | user | float | user_features | Days from the user's last rating to the split cutoff | `(cutoff - max(event_time)) in days` |
| `user_active_span_days` | user | int | user_features | Days between first and last rating | `max(event_time) - min(event_time)` |
| `user_preferred_genre` | user | string | user_features | Genre with the largest share of the user's liked items | `argmax(user_genre_share_*)` |
| `user_distinct_genres` | user | int | user_features | Number of genres the user has liked at least once | `count(user_genre_share_* > 0)` |
| `user_genre_share_<genre>` | user | float | user_genre_profile | Share of the user's liked items that carry the genre (rows sum to 1) | `sum(liked items with genre) / sum(liked genre tags)` |
| `item_popularity` | item | int | item_features | Number of train ratings the item received | `count(train rows of item)` |
| `item_avg_rating` | item | float | item_features | Mean rating of the item | `mean(rating)` |
| `item_bayes_rating` | item | float | item_features | Mean rating pulled towards the global mean for items with few ratings | `(n * item_mean + m * global_mean) / (n + m), m = prior strength` |
| `item_positive_ratio` | item | float | item_features | Share of the item's ratings that count as liked | `mean(rating >= relevance_threshold)` |
| `item_release_year` | item | int | item_features | Release year taken from the title | `regex (YYYY) in title, median if missing` |
| `item_age_years` | item | int | item_features | Age of the item at the split cutoff | `max(cutoff_year - release_year, 0)` |
| `item_days_since_last_interaction` | item | float | item_features | Days from the last rating of the item to the cutoff | `(cutoff - max(event_time)) in days` |
| `item_genre_count` | item | int | item_features | Number of genres on the item | `sum(genre_*)` |
| `genre_<genre>` | item | int8 | item_features | 1 when the item carries the genre | `genre in item.genres` |
| `rating_user_deviation` | interaction | float | interaction_features | How far the rating is from the user's own mean | `rating - user_avg_rating` |
| `rating_item_deviation` | interaction | float | interaction_features | How far the rating is from the item's mean | `rating - item_avg_rating` |
| `days_since_interaction` | interaction | float | interaction_features | Age of the interaction at the cutoff | `(cutoff - event_time) in days` |
| `recency_weight` | interaction | float | interaction_features | Exponential decay of the interaction age | `0.5 ** (days_since_interaction / half_life)` |
| `user_genre_affinity` | interaction | float | interaction_features | How well the item's genres match the user's liked genres (available at prediction time) | `mean over item genres of user_genre_share_<genre>` |
