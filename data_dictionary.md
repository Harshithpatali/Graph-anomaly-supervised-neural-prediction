# Data Dictionary

## Fraud_Data.csv

| Column | Description |
|---|---|
| `user_id` | User identifier. |
| `signup_time` | Account signup timestamp. |
| `purchase_time` | Transaction/purchase timestamp. |
| `purchase_value` | Transaction value. |
| `device_id` | Device identifier. |
| `source` | Acquisition/source category. |
| `browser` | Browser category. |
| `sex` | Recorded sex category. |
| `age` | User age. |
| `ip_address` | Numeric IP representation. |
| `class` | Fraud ground-truth label used for validation. |

## IpAddress_to_Country.csv

| Column | Description |
|---|---|
| `lower_bound_ip_address` | Lower bound of IP range. |
| `upper_bound_ip_address` | Upper bound of IP range. |
| `country` | Country associated with the IP range. |
