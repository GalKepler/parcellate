.. parcellate documentation master file

Welcome to parcellate's documentation!
======================================

``parcellate`` is a BIDS App for extracting regional statistics from scalar neuroimaging maps using atlas-based parcellation. It integrates with **CAT12** (VBM) and **QSIRecon** (dMRI) preprocessing pipelines and produces tidy TSV tables suitable for downstream analysis and quality control.

.. grid:: 1 2 2 2
   :gutter: 2
   :margin: 2 0 2 0

   .. grid-item-card:: Quick start
      :link: getting_started
      :link-type: doc
      :text-align: center

      Install the package, configure your environment, and run your first parcellation.

   .. grid-item-card:: CAT12 guide
      :link: cat12_guide
      :link-type: doc
      :text-align: center

      Process CAT12 VBM outputs: input layout, output format, masking, and TIV.

   .. grid-item-card:: QSIRecon guide
      :link: qsirecon_guide
      :link-type: doc
      :text-align: center

      Process QSIRecon diffusion outputs including 4D probabilistic atlases.

   .. grid-item-card:: API reference
      :link: api
      :link-type: doc
      :text-align: center

      Explore the VolumetricParcellator and the built-in statistical functions.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   getting_started

.. toctree::
   :maxdepth: 2
   :caption: Pipeline guides

   cat12_guide
   qsirecon_guide

.. toctree::
   :maxdepth: 2
   :caption: Reference

   cli_reference
   metrics_reference
   configuration
   usage
   troubleshooting
   api
