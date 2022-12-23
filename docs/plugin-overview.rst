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
      - custom/attribute-distances
      - application/zip
      - custom/entity-distances
      - application/zip

    * - :rspan:`2` `Costume Loader <https://github.com/UST-QuAntiL/qhana-plugin-runner/tree/main/plugins/costume_loader_pkg>`_
      - :rspan:`2` 
      - :rspan:`2` data-loading
      - :rspan:`2` 
      - :rspan:`2` 
      - :rspan:`2` 
      - entity/list
      - application/json
  
    * - entity/attribute-metadata
      - application/json
  
    * - custom/taxonomy-graphs
      - application/zip
       
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
      - custom/real-valued-entities
      - application/json
      - custom/real-valued-entities
      - application/json
  
    * - `JSON Visualization <https://github.com/UST-QuAntiL/qhana-plugin-runner/blob/main/plugins/json_visualization.py>`_
      - 
      - 
      - data
      - \*
      - application/json
      - custom/hello-world-output
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
      - custom/entity-distances
      - application/json
      - entity/vector
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
      - custom/nisq-analyzer-result
      - application/json
  
    * - :rspan:`2` `One-Hot Encoding <https://github.com/UST-QuAntiL/qhana-plugin-runner/blob/main/plugins/one-hot_encoding.py>`_
      - :rspan:`2` One-Hot Encoding
      - :rspan:`2` encoding, one-hot
      - entitiesUrl
      - entity/list
      - application/json
      - :rspan:`2` entity/vector
      - :rspan:`2` application/csv
  
    * - taxonomiesZipUrl
      - graph/taxonomy
      - application/zip
  
    * - entitiesMetadataUrl
      - entity/attribute-metadata
      - application/json  

    * - :rspan:`2` `Principle Component Analysis <https://github.com/UST-QuAntiL/qhana-plugin-runner/tree/main/plugins/pca>`_
      - :rspan:`2` normal, incremental, sparse and kernel PCA
      - :rspan:`2` dimension-reduction
      - :rspan:`2` entityPointsUrl
      - :rspan:`2` entity/vector
      - :rspan:`2` text/csv, application/json
      - custom/plot
      - text/html
  
    * - custom/pca-metadata
      - application/json
  
    * - entity/vector
      - text/csv
  
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
      - entity/vector
      - application/json
      - custom/clusters
      - application/json

    * - :rspan:`1` `Quantum Kernel Estimation <https://github.com/UST-QuAntiL/qhana-plugin-runner/tree/main/plugins/quantum_kernel_estimation>`_
      - :rspan:`1` Quantum Kernel Estimation with kernels from `Havlíček, V., Córcoles, A.D., Temme, K. et al. Supervised learning with quantum-enhanced feature spaces. Nature 567, 209–212 (2019). <https://doi.org/10.1038/s41586-019-0980-2>`
      and `Suzuki, Y., Yano, H., Gao, Q. et al. Analysis and synthesis of feature map for kernel-based quantum classifier. Quantum Mach. Intell. 2, 9 (2020). <https://doi.org/10.1007/s42484-020-00020-y>`
      - :rspan:`1` kernel, mapping
      - entityPointsUrl1
      - entity/vector
      - application/json
      - :rspan:`1` custom/kernel-matrix
      - :rspan:`1` application/json
    
    * - entityPointsUrl2
      - entity/vector
      - application/json

    * - :rspan:`1` `Qiskit Quantum Kernel Estimation <https://github.com/UST-QuAntiL/qhana-plugin-runner/tree/main/plugins/qiskit_quantum_kernel_estimation>`_
      - :rspan:`1` Qiskit Quantum Kernel using qiskit's feature maps (ZFeatureMap, ZZFeatureMap, PauliFeatureMap) using the proposed kernel from `Havlíček, V., Córcoles, A.D., Temme, K. et al. Supervised learning with quantum-enhanced feature spaces. Nature 567, 209–212 (2019). <https://www.nature.com/articles/s41586-019-0980-2>`
      - :rspan:`1` kernel, mapping
      - entityPointsUrl1
      - entity/vector
      - application/json
      - :rspan:`1` custom/kernel-matrix
      - :rspan:`1` application/json
    
    * - entityPointsUrl2
      - entity/vector
      - application/json
  
    * - :rspan:`3` `Qiskit Simulator <https://github.com/UST-QuAntiL/qhana-plugin-runner/blob/main/plugins/qiskit_simulator.py>`_
      - :rspan:`3` 
      - :rspan:`3` circuit-executor, qc-simulator, qiskit, qasm, qasm-2
      - :rspan:`1` circuit
      - :rspan:`1` executable/circuit
      - :rspan:`1` text/x-qasm
      - entity/vector
      - application/json
  
    * - entity/vector
      - application/json
  
    * - :rspan:`1` executionOptions
      - :rspan:`1` provenance/execution-options
      - :rspan:`1` text/csv, application/json, application/X-lines+json
      - provenance/trace
      - application/json
    
    * - provenance/execution-options
      - application/json
      
    * - :rspan:`1` `Sym Max Mean Attribute Comparer <https://github.com/UST-QuAntiL/qhana-plugin-runner/blob/main/plugins/sym_max_mean.py>`_
      - :rspan:`1` 
      - :rspan:`1` attribute-similarity-calculation
      - entitiesUrl
      - entity/list
      - application/json
      - :rspan:`1` custom/attribute-similarities
      - :rspan:`1` application/zip
  
    * - elementSimilaritiesUrl
      - custom/element-similarities
      - application/zip
  
    * - `Time Tanh Similarities <https://github.com/UST-QuAntiL/qhana-plugin-runner/blob/main/plugins/time_tanh.py>`_
      - 
      - similarity-calculation
      - entitiesUrl
      - entity/list
      - application/json
      - custom/element-similarities
      - application/zip
      
    * - `Transformers (Similarity to Distance) <https://github.com/UST-QuAntiL/qhana-plugin-runner/blob/main/plugins/transformers.py>`_
      - Linear/ Exponential/ Gaussian/ Polynomial/ Square Inverse
      - sim-to-dist
      - attributeSimilaritiesUrl
      - custom/attribute-similarities
      - application/zip
      - custom/attribute-distances
      - application/zip
      
    * - :rspan:`1` `Visualization <https://github.com/UST-QuAntiL/qhana-plugin-runner/blob/main/plugins/visualization.py>`_
      - :rspan:`1` 
      - :rspan:`1` visualization
      - entityPointsUrl
      - entity/numeric
      - application/json
      - :rspan:`1` 
      - :rspan:`1` 
  
    * - clustersUrl
      - custom/clusters
      - application/json

    * -  `Workflows <https://github.com/UST-QuAntiL/qhana-plugin-runner/tree/main/plugins/qiskit_quantum_kernel_estimation>`_
      - 
      - workflow, bpmn
      - 
      - 
      - 
      - 
      - 
  
    * - :rspan:`2` `Wu Palmer <https://github.com/UST-QuAntiL/qhana-plugin-runner/blob/main/plugins/wu_palmer.py>`_
      - :rspan:`2` Wu Palmer
      - :rspan:`2` similarity-calculation
      - entitiesUrl
      - entity/list
      - application/json
      - :rspan:`2` custom/element-similarities
      - :rspan:`2` application/zip
    
    * - entitiesMetadataUrl
      - entity/attribute-metadata
      - application/json

    * - taxonomiesZipUrl
      - graph/taxonomy
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
  