Plugin Overview
===============

.. flat-table::
    :header-rows: 2

    * - :rspan:`1` Name
      - :rspan:`1` Algorithm 
      - :rspan:`1` Tags 
      - :cspan:`2` Inputs
      - :cspan:`1` Outputs
    
    * - parameter
      - data type
      - content type
      - data type
      - content type
          
    * - `Aggregators <https://github.com/UST-QuAntiL/qhana-plugin-runner/blob/main/plugins/aggregators.py>`_
      - Mean/ Meadian/ Max/ Min Aggregators
      - aggregator
      - attributeDistancesUrl
      - attribute-distances
      - application/zip
      - entity-distances
      - application/zip

    * - :rspan:`2` `Costume Loader <https://github.com/UST-QuAntiL/qhana-plugin-runner/tree/main/plugins/costume_loader_pkg>`_
      - :rspan:`2` 
      - :rspan:`2` data-loading
      - :rspan:`2` 
      - :rspan:`2` 
      - :rspan:`2` 
      - raw
      - application/json
  
    * - attribute-metadata
      - application/json
  
    * - graphs
      - application/json
       
    * - `CSV Visualization <https://github.com/UST-QuAntiL/qhana-plugin-runner/blob/main/plugins/csv_visualization.py>`_
      - 
      - visualization
      - data
      - \*
      - text/csv
      - \*
      - text/html
      
    * - `Entity Filter <https://github.com/UST-QuAntiL/qhana-plugin-runner/blob/main/plugins/entity_filter.py>`_
      - 
      - data-loading
      - inputFileUrl
      - entity/list
      - application/json, text/csv
      - entity/list
      - application/json, text/csv
      
    * - `File Upload <https://github.com/UST-QuAntiL/qhana-plugin-runner/blob/main/plugins/file_upload.py>`_
      - 
      - data-loading
      - 
      - 
      - 
      - \*
      - \*
      
    * - `Hello World <https://github.com/UST-QuAntiL/qhana-plugin-runner/blob/main/plugins/hello_world.py>`_
      - 
      - hello-world
      - 
      - 
      - 
      - txt
      - text/plain
  
    * - `Hello World Multi-Step <https://github.com/UST-QuAntiL/qhana-plugin-runner/tree/main/plugins/hello_worl_multi_step>`_
      - 
      - hello-world, multistep
      - 
      - 
      - 
      - txt
      - text/plain
      
      
    * - `Hybrid Autoencoder <https://github.com/UST-QuAntiL/qhana-plugin-runner/tree/main/plugins/hybrid_ae_pkg>`_
      - `Quantum Autoencoder <https://arxiv.org/abs/1612.02806>`_, `QNN1+2 (Circuit A+B) <https://arxiv.org/abs/1612.02806>`_, `QNN3 <http://arxiv.org/abs/2011.00027>`_, `General Two-Qubit-Gate <https://arxiv.org/abs/quant-ph/0308006>`_
      - dimensionality-reduction
      - 
      - real-valued-entities
      - application/json
      - real-valued-entities
      - application/json
  
    * - `JSON Visualization <https://github.com/UST-QuAntiL/qhana-plugin-runner/blob/main/plugins/json_visualization.py>`_
      - 
      - 
      - data
      - \*
      - application/json
      - \*
      - text/html

    * - `Manual Classification <https://github.com/UST-QuAntiL/qhana-plugin-runner/tree/main/plugins/manual_classification>`_
      - 
      - data-annotation
      - inputFileUrl
      - entity/list
      - application/json, text/csv 
      - entity/list
      - application/json, text/csv
      
    * - `Multidimensional Scaling <https://github.com/UST-QuAntiL/qhana-plugin-runner/blob/main/plugins/mds.py>`_
      - `Metric or nonmetric MDS <https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html#sklearn.manifold.MDS>`_
      - dist-to-points
      - entityDistancesUrl
      - entity-distances
      - application/json
      - entity-points
      - application/json
      
    * - `Minio Storage <https://github.com/UST-QuAntiL/qhana-plugin-runner/blob/main/plugins/minio-storage.py>`_
      - 
      - 
      - 
      - 
      - 
      - 
      - 
      
    * - `NISQ Analyzer <https://github.com/UST-QuAntiL/qhana-plugin-runner/blob/main/plugins/nisq_analyzer.py>`_
      - 
      - nisq-analyzer
      - 
      - 
      - 
      - 
      - 
      
    * - `Principle Component Analysis <https://github.com/UST-QuAntiL/qhana-plugin-runner/blob/main/plugins/pca.py>`_
      - `PCA <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_
      - dimension-reduction
      - entityPointsUrl
      - entity-points
      - application/json
      - principle-components
      - application/json

    * - `Quantum k-Means <https://github.com/UST-QuAntiL/qhana-plugin-runner/tree/main/plugins/quantum_k_means>`_
      - `Quantum k-Means <https://arxiv.org/abs/1909.12183>`_ with custom adaptations for State-Preparation-Quantum-k-Means
      - points-to-clusters, k-means
      - entityPointsUrl
      - entity-points
      - application/json
      - clusters
      - application/json
      
    * - :rspan:`1` `Sym Max Mean Attribute Comparer <https://github.com/UST-QuAntiL/qhana-plugin-runner/blob/main/plugins/sym_max_mean.py>`_
      - :rspan:`1` 
      - :rspan:`1` attribute-similarity-calculation
      - entitiesUrl
      - entities
      - application/json
      - :rspan:`1` attribute-similarities
      - :rspan:`1` application/zip
  
    * - elementSimilaritiesUrl
      - element-similarities
      - application/zip
  
    * - `Time Tanh Similarities <https://github.com/UST-QuAntiL/qhana-plugin-runner/blob/main/plugins/time_tanh.py>`_
      - 
      - similarity-calculation
      - entitiesUrl
      - entities
      - application/json
      - element-similarities
      - application/zip
      
    * - `Transformers (Similarity to Distance) <https://github.com/UST-QuAntiL/qhana-plugin-runner/blob/main/plugins/transformers.py>`_
      - Linear/ Exponential/ Gaussian/ Polynomial/ Square Inverse
      - sim-to-dist
      - attributeSimilaritiesUrl
      - attribute-similarities
      - application/zip
      - attribute-distances
      - application/zip
      
    * - :rspan:`1` `Visualization <https://github.com/UST-QuAntiL/qhana-plugin-runner/blob/main/plugins/visualization.py>`_
      - :rspan:`1` 
      - :rspan:`1` visualization
      - entityPointsUrl
      - entity-points
      - application/json
      - :rspan:`1` 
      - :rspan:`1` 
  
    * - clustersUrl
      - clusters
      - application/json

    * - :rspan:`2` `Wu Palmer <https://github.com/UST-QuAntiL/qhana-plugin-runner/blob/main/plugins/wu_palmer.py>`_
      - :rspan:`2` Wu Palmer
      - :rspan:`2` similarity-calculation
      - entitiesUrl
      - entities
      - application/json
      - :rspan:`2` element-similarities
      - :rspan:`2` application/zip
    
    * - entitiesMetadataUrl
      - attribute-metadata
      - application/json

    * - taxonomiesZipUrl
      - taxonomy
      - application/zip
  
    * - :rspan:`1` `Zip Merger <https://github.com/UST-QuAntiL/qhana-plugin-runner/blob/main/plugins/zip_merger.py>`_
      - :rspan:`1` 
      - :rspan:`1` utility
      - zip1Url
      - any
      - application/zip
      - :rspan:`1` any
      - :rspan:`1` application/zip
  
    * - zip2Url
      - any
      - application/zip
  