# Security Predictor

The model needs to be downloaded from:

https://we.tl/t-Cxub1v8t0H

and stored in the folder `model-parameters`.

To run the API from the container:
```
source run_api.sh

```

## Example

Example to get a future job:

`curl -v http://127.0.0.1:8080/?prevjob=senior%20data%20analyst`

and will return a json with the probabilities (in %) for each predicted job:

```
[
  [
    "statistical analyst",
    76
  ],
  [
    "data analyst",
    8
  ],
  [
    "statistical programmer",
    6
  ],
  [
    "data analytics",
    6
  ],
  [
    "statistical analyst programmer",
    2
  ],
  [
    "data analytics specialist",
    2
  ]
]
```
