# Retrain the mem2mem model with only the rollout training instead of 50/50 split

We want to verify rather just the rollout training can also yield the same impressive results that the 50/50 split achived.
So run new experiment on the same code, but train only with the rollout trainig. Then do the rollout test with window=8 max_k = 64 and check rather it still has new perfect retention at high k.