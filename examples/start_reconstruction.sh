# pm.sample mode below
python3 run.py \
  --run BicyclusExample \
  --sample-parameters-file parameters/sample_parameters.json \
  --true-parameters-file parameters/true_parameters.json \
  --algorithm default \
  --cores 4 \
  --chains 4 \
  --iterations 1 \
  --samples 200 \
  --tune 100 \
  --rel-sigma 0.5 \
  --debug  # Print log to CLI, don't store log in a separate file.

# pm.iter_sample mode below
# python3 run.py \
#   --run BicyclusExample \
#   --sample-parameters-file parameters/sample_parameters.json \
#   --true-parameters-file parameters/true_parameters.json \
#   --algorithm default \
#   --iter-sample 50 \
#   --iterations 1 \
#   --samples 200 \
#   --tune 100 \
#   --rel-sigma 0.5 \
#   --debug  # Print log to CLI, don't store log in a separate file.
