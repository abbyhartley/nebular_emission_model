# nebular_emission_model
Empirical model for galaxy nebular emission:

Our goal is to train a normalizing flow conditioned on galaxy stellar mass and H-alpha luminosity to predict the covariant distribution of 8 bright optical emission lines. The block neural autoregressive flow is constructed using FlowJAX, and it is trained on data from DESI BGS and SDSS MGS.
