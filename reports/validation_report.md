# Data validation report (movietweetings)

## Rows per table

| table | rows in | valid | rejected | reject ratio |
|---|---:|---:|---:|---:|
| users | 16,556 | 16,554 | 2 | 0.01% |
| items | 10,510 | 10,507 | 3 | 0.03% |
| interactions | 100,200 | 99,000 | 1,200 | 1.20% |

## Checks

| table | check | severity | failed rows | rule |
|---|---|---|---:|---|
| users | `missing_user_id` | error | 0 | user_id must not be null |
| users | `user_id_not_numeric` | error | 0 | user_id must be an integer string |
| users | `duplicate_user_id` | error | 2 | user_id must be unique (first copy kept) |
| items | `missing_item_id` | error | 0 | item_id must not be null |
| items | `item_id_not_numeric` | error | 0 | item_id must be an integer string |
| items | `duplicate_item_id` | error | 1 | item_id must be unique (first copy kept) |
| items | `missing_title` | error | 2 | title must not be null |
| items | `missing_genres` | warning | 68 | genres empty, filled with 'Unknown' later |
| items | `missing_release_year` | warning | 1 | no year in title, imputed later |
| interactions | `missing_user_id` | error | 0 | user_id must not be null |
| interactions | `missing_item_id` | error | 0 | item_id must not be null |
| interactions | `missing_rating` | error | 200 | rating must not be null |
| interactions | `missing_timestamp` | error | 0 | timestamp must not be null |
| interactions | `user_id_not_numeric` | error | 0 | user_id must be an integer string |
| interactions | `item_id_not_numeric` | error | 0 | item_id must be an integer string |
| interactions | `rating_not_numeric` | error | 200 | rating must be numeric (wrong data type) |
| interactions | `timestamp_not_integer` | error | 0 | timestamp must be an integer unix time (wrong data type) |
| interactions | `rating_out_of_range` | error | 200 | rating must be in [0, 10] |
| interactions | `invalid_timestamp` | error | 200 | timestamp must be between 2010-01-01 and now |
| interactions | `orphan_user` | error | 200 | user_id must exist in users (referential integrity) |
| interactions | `orphan_item` | error | 200 | item_id must exist in items (referential integrity) |
| interactions | `duplicate_user_item` | warning | 200 | one rating per user and item; older copy dropped in preparation |

Severity `error`: row moved to `data/errors/`. Severity `warning`: row kept, fixed in preparation.

## Injected defects

Ingestion adds a small logged set of defects (`data/raw/injected_faults.csv`) to prove the checks work.

Caught 1406 of 1406.

| defect | injected | caught |
|---|---:|---:|
| duplicate_item_id | 1 | 1 |
| duplicate_user_id | 2 | 2 |
| duplicate_user_item | 200 | 200 |
| missing_genres | 1 | 1 |
| missing_rating | 200 | 200 |
| missing_title | 2 | 2 |
| orphan_item | 200 | 200 |
| orphan_user | 200 | 200 |
| rating_not_numeric | 200 | 200 |
| rating_out_of_range | 200 | 200 |
| timestamp_in_future | 200 | 200 |
