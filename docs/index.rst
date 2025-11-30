.. parcellate documentation master file

Welcome to parcellate's documentation!
======================================

``parcellate`` is a lightweight toolkit for computing parcel-wise statistics from 3D neuroimaging volumes. It pairs a volumetric atlas with scalar maps, handles resampling, and produces tidy tables of region-level measurements suitable for downstream analysis and quality control.

.. grid:: 1 2 2 2
   :gutter: 2
   :margin: 2 0 2 0

   .. grid-item-card:: Quick start
      :link: getting_started
      :link-type: doc
      :text-align: center

      Install the package, configure your environment, and run your first parcellation.

   .. grid-item-card:: API reference
      :link: api
      :link-type: doc
      :text-align: center

      Explore the VolumetricParcellator and the built-in statistical functions.

   .. grid-item-card:: Usage guide
      :link: usage
      :link-type: doc
      :text-align: center

      Learn how to customize atlases, lookup tables, and summary statistics.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   getting_started
   usage

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
