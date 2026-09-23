# Agent 模型与参数执行覆盖核对

核对日期：2026-09-05。由当前源码签名和 Agent 参数路由生成；不代表已对全部模型完成真实训练。

12 个模型均已注册。params 内不带前缀的是统一入口选项；model./trainer./fit./prepare./main./config./pipeline./word2vec./embedding. 分别连接明确接收端。

所有执行仍通过独立宿主确认。本地适配器支持扩展参数；旧 Go worker 未登记扩展协议时会明确拒绝，不丢弃参数。

数据/任务绑定字段由宿主提供；原实现无效或覆盖的字段明确列出，不宣称可以调整。DTM 的 time_slices 只能与既有分箱相符，不做隐式重分桶；分布式 rank/world_size 属于启动器，当前本地启动器为单进程 CPU 或单张 CUDA。

统一 CLI 未暴露不等于 Agent 不可调：例如 NVDM/GSM 的 model.dropout、ProdLDA 的 model.variance、ETM 的 model.train_embeddings 已通过原类 API 接入。使用下方对应入口章节核实；请勿把旧清单中的统一 CLI 限制当作当前 Agent 限制。

## run_pipeline.py

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| run_pipeline.py | dataset | 宿主管理 | host dataset binding |
| run_pipeline.py | models | 宿主管理 | plan.modelId |
| run_pipeline.py | mode | 可配置 | theta: params.mode |
| run_pipeline.py | num_topics | 可配置 | lda: params.num_topics; stm: params.num_topics; btm: params.num_topics; etm: params.num_topics; ctm: params.num_topics; dtm: params.num_topics; nvdm: params.num_topics; gsm: params.num_topics; prodlda: params.num_topics; bertopic: params.num_topics; theta: params.num_topics |
| run_pipeline.py | vocab_size | 可配置 | lda: params.vocab_size; lda: params.prepare.vocab_size; hdp: params.vocab_size; hdp: params.prepare.vocab_size; stm: params.vocab_size; stm: params.prepare.vocab_size; btm: params.vocab_size; btm: params.prepare.vocab_size; etm: params.vocab_size; etm: params.prepare.vocab_size; ctm: params.vocab_size; ctm: params.prepare.vocab_size; dtm: params.vocab_size; dtm: params.prepare.vocab_size; nvdm: params.vocab_size; nvdm: params.prepare.vocab_size; gsm: params.vocab_size; gsm: params.prepare.vocab_size; prodlda: params.vocab_size; prodlda: params.prepare.vocab_size; bertopic: params.vocab_size; bertopic: params.prepare.vocab_size; theta: params.vocab_size; theta: params.prepare.vocab_size |
| run_pipeline.py | epochs | 可配置 | etm: params.epochs; ctm: params.epochs; dtm: params.epochs; nvdm: params.epochs; gsm: params.epochs; prodlda: params.epochs; theta: params.epochs |
| run_pipeline.py | batch_size | 可配置 | lda: params.prepare.batch_size; hdp: params.prepare.batch_size; stm: params.prepare.batch_size; btm: params.prepare.batch_size; etm: params.batch_size; etm: params.prepare.batch_size; ctm: params.batch_size; ctm: params.prepare.batch_size; dtm: params.batch_size; dtm: params.prepare.batch_size; nvdm: params.batch_size; nvdm: params.prepare.batch_size; gsm: params.batch_size; gsm: params.prepare.batch_size; prodlda: params.batch_size; prodlda: params.prepare.batch_size; bertopic: params.prepare.batch_size; theta: params.batch_size; theta: params.prepare.batch_size |
| run_pipeline.py | hidden_dim | 可配置 | etm: params.hidden_dim; ctm: params.hidden_dim; dtm: params.hidden_dim; nvdm: params.hidden_dim; gsm: params.hidden_dim; prodlda: params.hidden_dim; theta: params.hidden_dim |
| run_pipeline.py | learning_rate | 可配置 | etm: params.learning_rate; ctm: params.learning_rate; dtm: params.learning_rate; nvdm: params.learning_rate; gsm: params.learning_rate; prodlda: params.learning_rate; theta: params.learning_rate |
| run_pipeline.py | kl_start | 可配置 | theta: params.kl_start |
| run_pipeline.py | kl_end | 可配置 | theta: params.kl_end |
| run_pipeline.py | kl_warmup | 可配置 | theta: params.kl_warmup |
| run_pipeline.py | patience | 可配置 | etm: params.patience; ctm: params.patience; theta: params.patience |
| run_pipeline.py | no_early_stopping | 可配置 | theta: params.no_early_stopping |
| run_pipeline.py | skip_train | 宿主管理 | results_read for existing job |
| run_pipeline.py | skip_eval | 可配置 | lda: params.skip_eval; hdp: params.skip_eval; stm: params.skip_eval; btm: params.skip_eval; etm: params.skip_eval; ctm: params.skip_eval; dtm: params.skip_eval; nvdm: params.skip_eval; gsm: params.skip_eval; prodlda: params.skip_eval; bertopic: params.skip_eval; theta: params.skip_eval |
| run_pipeline.py | skip_viz | 可配置 | lda: params.skip_viz; hdp: params.skip_viz; stm: params.skip_viz; btm: params.skip_viz; etm: params.skip_viz; ctm: params.skip_viz; dtm: params.skip_viz; nvdm: params.skip_viz; gsm: params.skip_viz; prodlda: params.skip_viz; bertopic: params.skip_viz; theta: params.skip_viz |
| run_pipeline.py | gpu | 宿主管理 | plan.device |
| run_pipeline.py | language | 可配置 | lda: params.language; hdp: params.language; stm: params.language; btm: params.language; etm: params.language; ctm: params.language; dtm: params.language; nvdm: params.language; gsm: params.language; prodlda: params.language; bertopic: params.language; theta: params.language |
| run_pipeline.py | model_size | 可配置 | theta: params.model_size |
| run_pipeline.py | embedding_provider | 可配置 | theta: params.embedding_provider |
| run_pipeline.py | embedding_cloud_provider | 可配置 | theta: params.embedding_cloud_provider |
| run_pipeline.py | embedding_model | 可配置 | theta: params.embedding_model |
| run_pipeline.py | embedding_api_base | 可配置 | theta: params.embedding_api_base |
| run_pipeline.py | embedding_api_key_env | 可配置 | theta: params.embedding_api_key_env |
| run_pipeline.py | embedding_dimensions | 可配置 | theta: params.embedding_dimensions |
| run_pipeline.py | check_only | 宿主管理 | runtime_check / training_prepare |
| run_pipeline.py | prepare | 宿主管理 | approved preprocessing stage |
| run_pipeline.py | max_iter | 可配置 | lda: params.max_iter; stm: params.max_iter |
| run_pipeline.py | max_topics | 可配置 | hdp: params.max_topics |
| run_pipeline.py | n_iter | 可配置 | btm: params.n_iter |
| run_pipeline.py | alpha | 可配置 | hdp: params.alpha; btm: params.alpha |
| run_pipeline.py | beta | 可配置 | btm: params.beta |
| run_pipeline.py | inference_type | 可配置 | ctm: params.inference_type |
| run_pipeline.py | dropout | 可配置 | etm: params.dropout |
| run_pipeline.py | num_layers | 可配置 | ctm: params.num_layers |
| run_pipeline.py | embedding_dim | 可配置 | etm: params.embedding_dim; dtm: params.embedding_dim |
| run_pipeline.py | n_neighbors | 可配置 | bertopic: params.n_neighbors |
| run_pipeline.py | n_components | 可配置 | bertopic: params.n_components |
| run_pipeline.py | min_cluster_size | 可配置 | bertopic: params.min_cluster_size |
| run_pipeline.py | min_samples | 可配置 | bertopic: params.min_samples |
| run_pipeline.py | top_n_words | 可配置 | bertopic: params.top_n_words |
| run_pipeline.py | random_state | 可配置 | bertopic: params.random_state |
| run_pipeline.py | data_exp | 宿主管理 | prepared job artifact |
| run_pipeline.py | exp_name | 宿主管理 | job identity |
| run_pipeline.py | user_id | 宿主管理 | host isolation |
| run_pipeline.py | workspace_dir | 宿主管理 | job workspace |
| run_pipeline.py | force | 宿主管理 | fresh isolated job workspace |
| run_pipeline.py | task_name | 宿主管理 | job identity |
| run_pipeline.py | lang | 可配置 | lda: params.lang; hdp: params.lang; stm: params.lang; btm: params.lang; etm: params.lang; ctm: params.lang; dtm: params.lang; nvdm: params.lang; gsm: params.lang; prodlda: params.lang; bertopic: params.lang; theta: params.lang |

## prepare_data.py

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| prepare_data.py | dataset | 宿主管理 | host dataset binding |
| prepare_data.py | model | 宿主管理 | derived preparation kind |
| prepare_data.py | model_size | 可配置 | theta: params.model_size |
| prepare_data.py | embedding_provider | 可配置 | theta: params.embedding_provider |
| prepare_data.py | embedding_cloud_provider | 可配置 | theta: params.embedding_cloud_provider |
| prepare_data.py | embedding_model | 可配置 | theta: params.embedding_model |
| prepare_data.py | embedding_api_base | 可配置 | theta: params.embedding_api_base |
| prepare_data.py | embedding_api_key_env | 可配置 | theta: params.embedding_api_key_env |
| prepare_data.py | embedding_dimensions | 可配置 | theta: params.embedding_dimensions |
| prepare_data.py | mode | 可配置 | theta: params.mode |
| prepare_data.py | vocab_size | 可配置 | lda: params.vocab_size; lda: params.prepare.vocab_size; hdp: params.vocab_size; hdp: params.prepare.vocab_size; stm: params.vocab_size; stm: params.prepare.vocab_size; btm: params.vocab_size; btm: params.prepare.vocab_size; etm: params.vocab_size; etm: params.prepare.vocab_size; ctm: params.vocab_size; ctm: params.prepare.vocab_size; dtm: params.vocab_size; dtm: params.prepare.vocab_size; nvdm: params.vocab_size; nvdm: params.prepare.vocab_size; gsm: params.vocab_size; gsm: params.prepare.vocab_size; prodlda: params.vocab_size; prodlda: params.prepare.vocab_size; bertopic: params.vocab_size; bertopic: params.prepare.vocab_size; theta: params.vocab_size; theta: params.prepare.vocab_size |
| prepare_data.py | batch_size | 可配置 | lda: params.prepare.batch_size; hdp: params.prepare.batch_size; stm: params.prepare.batch_size; btm: params.prepare.batch_size; etm: params.batch_size; etm: params.prepare.batch_size; ctm: params.batch_size; ctm: params.prepare.batch_size; dtm: params.batch_size; dtm: params.prepare.batch_size; nvdm: params.batch_size; nvdm: params.prepare.batch_size; gsm: params.batch_size; gsm: params.prepare.batch_size; prodlda: params.batch_size; prodlda: params.prepare.batch_size; bertopic: params.prepare.batch_size; theta: params.batch_size; theta: params.prepare.batch_size |
| prepare_data.py | max_length | 可配置 | ctm: params.prepare.max_length; dtm: params.prepare.max_length; bertopic: params.prepare.max_length; theta: params.prepare.max_length |
| prepare_data.py | bow_only | 可配置 | lda: params.prepare.bow_only; hdp: params.prepare.bow_only; stm: params.prepare.bow_only; btm: params.prepare.bow_only; etm: params.prepare.bow_only; ctm: params.prepare.bow_only; dtm: params.prepare.bow_only; nvdm: params.prepare.bow_only; gsm: params.prepare.bow_only; prodlda: params.prepare.bow_only; bertopic: params.prepare.bow_only; theta: params.prepare.bow_only |
| prepare_data.py | skip_sbert | 可配置 | lda: params.prepare.skip_sbert; hdp: params.prepare.skip_sbert; stm: params.prepare.skip_sbert; btm: params.prepare.skip_sbert; etm: params.prepare.skip_sbert; ctm: params.prepare.skip_sbert; dtm: params.prepare.skip_sbert; nvdm: params.prepare.skip_sbert; gsm: params.prepare.skip_sbert; prodlda: params.prepare.skip_sbert; bertopic: params.prepare.skip_sbert |
| prepare_data.py | with_time | 可配置 | lda: params.prepare.with_time; hdp: params.prepare.with_time; stm: params.prepare.with_time; btm: params.prepare.with_time; etm: params.prepare.with_time; ctm: params.prepare.with_time; dtm: params.prepare.with_time; nvdm: params.prepare.with_time; gsm: params.prepare.with_time; prodlda: params.prepare.with_time; bertopic: params.prepare.with_time |
| prepare_data.py | check_only | 宿主管理 | runtime_check / training_prepare |
| prepare_data.py | gpu | 宿主管理 | plan.device |
| prepare_data.py | clean | 可配置 | lda: params.prepare.clean; hdp: params.prepare.clean; stm: params.prepare.clean; btm: params.prepare.clean; etm: params.prepare.clean; ctm: params.prepare.clean; dtm: params.prepare.clean; nvdm: params.prepare.clean; gsm: params.prepare.clean; prodlda: params.prepare.clean; bertopic: params.prepare.clean; theta: params.prepare.clean |
| prepare_data.py | raw_input | 宿主管理 | approved dataset |
| prepare_data.py | language | 可配置 | lda: params.language; hdp: params.language; stm: params.language; btm: params.language; etm: params.language; ctm: params.language; dtm: params.language; nvdm: params.language; gsm: params.language; prodlda: params.language; bertopic: params.language; theta: params.language |
| prepare_data.py | time_column | 宿主管理 | plan.timeColumn |
| prepare_data.py | time_slices | 可配置 | dtm: params.prepare.time_slices |
| prepare_data.py | covariate_columns | 宿主管理 | plan.covariates |
| prepare_data.py | label_col | 宿主管理 | plan.labelColumn |
| prepare_data.py | exp_name | 宿主管理 | job identity |
| prepare_data.py | user_id | 宿主管理 | host isolation |
| prepare_data.py | output_dir | 宿主管理 | job workspace |
| prepare_data.py | force | 宿主管理 | fresh isolated job workspace |

## config.py

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| config.py | timestamp | 宿主管理 | selected result job |
| config.py | no_wordcloud | 原实现无效 | 声明但无消费端，不伪装为可生效参数 |
| config.py | input | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| config.py | output | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| config.py | language | 可配置 | lda: params.language; hdp: params.language; stm: params.language; btm: params.language; etm: params.language; ctm: params.language; dtm: params.language; nvdm: params.language; gsm: params.language; prodlda: params.language; bertopic: params.language; theta: params.language |
| config.py | dataset | 宿主管理 | host dataset binding |
| config.py | mode | 可配置 | theta: params.mode |
| config.py | config | 宿主管理 | config.* structured approved parameters |
| config.py | gpu | 宿主管理 | plan.device |
| config.py | dev | 可配置 | theta: params.main.dev |
| config.py | num_topics | 可配置 | lda: params.num_topics; stm: params.num_topics; btm: params.num_topics; etm: params.num_topics; ctm: params.num_topics; dtm: params.num_topics; nvdm: params.num_topics; gsm: params.num_topics; prodlda: params.num_topics; bertopic: params.num_topics; theta: params.num_topics |
| config.py | vocab_size | 可配置 | lda: params.vocab_size; lda: params.prepare.vocab_size; hdp: params.vocab_size; hdp: params.prepare.vocab_size; stm: params.vocab_size; stm: params.prepare.vocab_size; btm: params.vocab_size; btm: params.prepare.vocab_size; etm: params.vocab_size; etm: params.prepare.vocab_size; ctm: params.vocab_size; ctm: params.prepare.vocab_size; dtm: params.vocab_size; dtm: params.prepare.vocab_size; nvdm: params.vocab_size; nvdm: params.prepare.vocab_size; gsm: params.vocab_size; gsm: params.prepare.vocab_size; prodlda: params.vocab_size; prodlda: params.prepare.vocab_size; bertopic: params.vocab_size; bertopic: params.prepare.vocab_size; theta: params.vocab_size; theta: params.prepare.vocab_size |
| config.py | hidden_dim | 可配置 | etm: params.hidden_dim; ctm: params.hidden_dim; dtm: params.hidden_dim; nvdm: params.hidden_dim; gsm: params.hidden_dim; prodlda: params.hidden_dim; theta: params.hidden_dim |
| config.py | epochs | 可配置 | etm: params.epochs; ctm: params.epochs; dtm: params.epochs; nvdm: params.epochs; gsm: params.epochs; prodlda: params.epochs; theta: params.epochs |
| config.py | batch_size | 可配置 | lda: params.prepare.batch_size; hdp: params.prepare.batch_size; stm: params.prepare.batch_size; btm: params.prepare.batch_size; etm: params.batch_size; etm: params.prepare.batch_size; ctm: params.batch_size; ctm: params.prepare.batch_size; dtm: params.batch_size; dtm: params.prepare.batch_size; nvdm: params.batch_size; nvdm: params.prepare.batch_size; gsm: params.batch_size; gsm: params.prepare.batch_size; prodlda: params.batch_size; prodlda: params.prepare.batch_size; bertopic: params.prepare.batch_size; theta: params.batch_size; theta: params.prepare.batch_size |
| config.py | learning_rate | 可配置 | etm: params.learning_rate; ctm: params.learning_rate; dtm: params.learning_rate; nvdm: params.learning_rate; gsm: params.learning_rate; prodlda: params.learning_rate; theta: params.learning_rate |
| config.py | stage1_epochs | 可配置 | theta: params.main.stage1_epochs |
| config.py | stage2_epochs | 可配置 | theta: params.main.stage2_epochs |
| config.py | lora_r | 可配置 | theta: params.main.lora_r |
| config.py | lora_alpha | 可配置 | theta: params.main.lora_alpha |
| config.py | lora_dropout | 可配置 | theta: params.main.lora_dropout |
| config.py | kl_start | 可配置 | theta: params.kl_start |
| config.py | kl_end | 可配置 | theta: params.kl_end |
| config.py | kl_warmup | 可配置 | theta: params.kl_warmup |
| config.py | no_early_stopping | 可配置 | theta: params.no_early_stopping |
| config.py | patience | 可配置 | etm: params.patience; ctm: params.patience; theta: params.patience |
| config.py | train_word_embeddings | 可配置 | theta: params.main.train_word_embeddings |
| config.py | no_train_word_embeddings | 可配置 | theta: params.main.no_train_word_embeddings |
| config.py | enable_temporal | 可配置 | theta: params.main.enable_temporal |
| config.py | timestamp_column | 宿主管理 | plan.timeColumn |
| config.py | model_size | 可配置 | theta: params.model_size |
| config.py | embedding_provider | 可配置 | theta: params.embedding_provider |
| config.py | embedding_cloud_provider | 可配置 | theta: params.embedding_cloud_provider |
| config.py | embedding_model | 可配置 | theta: params.embedding_model |
| config.py | embedding_api_base | 可配置 | theta: params.embedding_api_base |
| config.py | embedding_api_key_env | 可配置 | theta: params.embedding_api_key_env |
| config.py | embedding_dimensions | 可配置 | theta: params.embedding_dimensions |
| config.py | skip_viz | 可配置 | lda: params.skip_viz; hdp: params.skip_viz; stm: params.skip_viz; btm: params.skip_viz; etm: params.skip_viz; ctm: params.skip_viz; dtm: params.skip_viz; nvdm: params.skip_viz; gsm: params.skip_viz; prodlda: params.skip_viz; bertopic: params.skip_viz; theta: params.skip_viz |
| config.py | skip_eval | 可配置 | lda: params.skip_eval; hdp: params.skip_eval; stm: params.skip_eval; btm: params.skip_eval; etm: params.skip_eval; ctm: params.skip_eval; dtm: params.skip_eval; nvdm: params.skip_eval; gsm: params.skip_eval; prodlda: params.skip_eval; bertopic: params.skip_eval; theta: params.skip_eval |
| config.py | data_exp | 宿主管理 | prepared job artifact |
| config.py | train_exp | 宿主管理 | job artifact |
| config.py | output_base_dir | 宿主管理 | job result directory |
| config.py | num_workers | 可配置 | theta: params.main.num_workers |
| config.py | no_pin_memory | 可配置 | theta: params.main.no_pin_memory |
| config.py | no_persistent_workers | 可配置 | theta: params.main.no_persistent_workers |
| config.py | label_col | 宿主管理 | plan.labelColumn |

## main.py

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| main.py | local_rank | 宿主管理 | launcher-owned |
| main.py | world_size | 宿主管理 | launcher-owned |

## ModelConfig

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| ModelConfig | num_topics | 可配置 | theta: params.num_topics |
| ModelConfig | hidden_dim | 可配置 | theta: params.hidden_dim |
| ModelConfig | doc_embedding_dim | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| ModelConfig | word_embedding_dim | 可配置 | theta: params.config.word_embedding_dim |
| ModelConfig | encoder_dropout | 可配置 | theta: params.config.encoder_dropout |
| ModelConfig | encoder_activation | 可配置 | theta: params.config.encoder_activation |
| ModelConfig | train_word_embeddings | 可配置 | theta: params.config.train_word_embeddings |
| ModelConfig | epochs | 可配置 | theta: params.epochs |
| ModelConfig | batch_size | 可配置 | theta: params.batch_size |
| ModelConfig | learning_rate | 可配置 | theta: params.learning_rate |
| ModelConfig | weight_decay | 可配置 | theta: params.config.weight_decay |
| ModelConfig | stage1_epochs | 可配置 | theta: params.config.stage1_epochs |
| ModelConfig | stage2_epochs | 可配置 | theta: params.config.stage2_epochs |
| ModelConfig | stage1_lr | 可配置 | theta: params.config.stage1_lr |
| ModelConfig | stage2_lr | 可配置 | theta: params.config.stage2_lr |
| ModelConfig | lora_r | 可配置 | theta: params.config.lora_r |
| ModelConfig | lora_alpha | 可配置 | theta: params.config.lora_alpha |
| ModelConfig | lora_dropout | 可配置 | theta: params.config.lora_dropout |
| ModelConfig | contrastive_temp | 可配置 | theta: params.config.contrastive_temp |
| ModelConfig | kl_start | 可配置 | theta: params.kl_start |
| ModelConfig | kl_end | 可配置 | theta: params.kl_end |
| ModelConfig | kl_warmup_epochs | 映射 | params.kl_warmup |
| ModelConfig | early_stopping | 可配置 | theta: params.config.early_stopping |
| ModelConfig | patience | 可配置 | theta: params.patience |
| ModelConfig | min_delta | 可配置 | theta: params.config.min_delta |
| ModelConfig | use_scheduler | 可配置 | theta: params.config.use_scheduler |
| ModelConfig | scheduler_patience | 可配置 | theta: params.config.scheduler_patience |
| ModelConfig | scheduler_factor | 可配置 | theta: params.config.scheduler_factor |
| ModelConfig | train_ratio | 可配置 | theta: params.config.train_ratio |
| ModelConfig | val_ratio | 可配置 | theta: params.config.val_ratio |
| ModelConfig | test_ratio | 可配置 | theta: params.config.test_ratio |
| ModelConfig | num_workers | 可配置 | theta: params.config.num_workers |
| ModelConfig | pin_memory | 可配置 | theta: params.config.pin_memory |
| ModelConfig | persistent_workers | 可配置 | theta: params.config.persistent_workers |
| ModelConfig | prefetch_factor | 可配置 | theta: params.config.prefetch_factor |

## EmbeddingConfig

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| EmbeddingConfig | mode | 可配置 | theta: params.mode |
| EmbeddingConfig | embedding_dim | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| EmbeddingConfig | model_path | 可配置 | theta: params.embedding.model_path |
| EmbeddingConfig | output_dir | 宿主管理 | job workspace |
| EmbeddingConfig | batch_size | 可配置 | theta: params.prepare.batch_size |
| EmbeddingConfig | max_length | 可配置 | theta: params.prepare.max_length |
| EmbeddingConfig | provider | 可配置 | theta: params.embedding_provider |
| EmbeddingConfig | cloud_provider | 可配置 | theta: params.embedding_cloud_provider |
| EmbeddingConfig | model | 可配置 | theta: params.embedding_model |
| EmbeddingConfig | api_base | 可配置 | theta: params.embedding_api_base |
| EmbeddingConfig | api_key_env | 可配置 | theta: params.embedding_api_key_env |
| EmbeddingConfig | dimensions | 可配置 | theta: params.embedding_dimensions |
| EmbeddingConfig | normalize | 可配置 | theta: params.embedding.normalize |

## PipelineConfig

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| PipelineConfig | seed | 可配置 | theta: params.pipeline.seed |

## lda.__init__

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| lda.__init__ | vocab_size | 可配置 | lda: params.vocab_size |
| lda.__init__ | num_topics | 可配置 | lda: params.num_topics |
| lda.__init__ | alpha | 可配置 | lda: params.model.alpha |
| lda.__init__ | eta | 可配置 | lda: params.model.eta |
| lda.__init__ | max_iter | 可配置 | lda: params.max_iter |
| lda.__init__ | learning_method | 可配置 | lda: params.model.learning_method |
| lda.__init__ | random_state | 可配置 | lda: params.model.random_state |
| lda.__init__ | n_jobs | 可配置 | lda: params.model.n_jobs |
| lda.__init__ | dev_mode | 可配置 | lda: params.model.dev_mode |
| lda.__init__ | doc_embedding_dim | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| lda.__init__ | word_embedding_dim | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| lda.__init__ | word_embeddings | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| lda.__init__ | train_word_embeddings | 兼容占位 | 本模型不消费词嵌入冻结选项 |

## lda.fit

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| lda.fit | bow_matrix | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |

## hdp.__init__

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| hdp.__init__ | vocab_size | 可配置 | hdp: params.vocab_size |
| hdp.__init__ | max_topics | 可配置 | hdp: params.max_topics |
| hdp.__init__ | alpha | 可配置 | hdp: params.alpha |
| hdp.__init__ | gamma | 可配置 | hdp: params.model.gamma |
| hdp.__init__ | kappa | 可配置 | hdp: params.model.kappa |
| hdp.__init__ | tau | 可配置 | hdp: params.model.tau |
| hdp.__init__ | K | 可配置 | hdp: params.model.K |
| hdp.__init__ | T | 原实现覆盖 | 使用 params.max_topics；原 T 被 max_topics 覆盖 |
| hdp.__init__ | random_state | 可配置 | hdp: params.model.random_state |

## hdp.fit

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| hdp.fit | bow_matrix | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| hdp.fit | vocab | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |

## stm.__init__

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| stm.__init__ | vocab_size | 可配置 | stm: params.vocab_size |
| stm.__init__ | num_topics | 可配置 | stm: params.num_topics |
| stm.__init__ | max_iter | 可配置 | stm: params.max_iter |
| stm.__init__ | random_state | 可配置 | stm: params.model.random_state |

## stm.fit

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| stm.fit | bow_matrix | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| stm.fit | covariates | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| stm.fit | covariate_names | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| stm.fit | vocab | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| stm.fit | dataset | 宿主管理 | host dataset binding |

## btm.__init__

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| btm.__init__ | vocab_size | 可配置 | btm: params.vocab_size |
| btm.__init__ | num_topics | 可配置 | btm: params.num_topics |
| btm.__init__ | alpha | 可配置 | btm: params.alpha |
| btm.__init__ | beta | 可配置 | btm: params.beta |
| btm.__init__ | n_iter | 可配置 | btm: params.n_iter |
| btm.__init__ | window_size | 可配置 | btm: params.model.window_size |
| btm.__init__ | max_doc_words | 可配置 | btm: params.model.max_doc_words |
| btm.__init__ | random_state | 可配置 | btm: params.model.random_state |

## btm.fit

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| btm.fit | bow_matrix | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| btm.fit | vocab | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| btm.fit | verbose | 可配置 | btm: params.fit.verbose |

## etm.__init__

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| etm.__init__ | vocab_size | 可配置 | etm: params.vocab_size |
| etm.__init__ | num_topics | 可配置 | etm: params.num_topics |
| etm.__init__ | embedding_dim | 可配置 | etm: params.embedding_dim |
| etm.__init__ | hidden_dim | 可配置 | etm: params.hidden_dim |
| etm.__init__ | dropout | 可配置 | etm: params.dropout |
| etm.__init__ | activation | 可配置 | etm: params.model.activation |
| etm.__init__ | word_embeddings | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| etm.__init__ | train_embeddings | 可配置 | etm: params.model.train_embeddings |
| etm.__init__ | kl_weight | 原训练器覆盖 | ETM 训练器自算 KL 退火，构造参数不控制训练器目标；不接收无效覆盖 |
| etm.__init__ | dev_mode | 可配置 | etm: params.model.dev_mode |
| etm.__init__ | doc_embedding_dim | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| etm.__init__ | word_embedding_dim | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |

## ctm.__init__

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| ctm.__init__ | vocab_size | 可配置 | ctm: params.vocab_size |
| ctm.__init__ | num_topics | 可配置 | ctm: params.num_topics |
| ctm.__init__ | doc_embedding_dim | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| ctm.__init__ | hidden_sizes | 可配置 | ctm: params.model.hidden_sizes |
| ctm.__init__ | activation | 可配置 | ctm: params.model.activation |
| ctm.__init__ | dropout | 可配置 | ctm: params.model.dropout |
| ctm.__init__ | model_type | 可配置 | ctm: params.model.model_type |
| ctm.__init__ | inference_type | 可配置 | ctm: params.inference_type |
| ctm.__init__ | learn_priors | 可配置 | ctm: params.model.learn_priors |
| ctm.__init__ | kl_weight | 可配置 | ctm: params.model.kl_weight |
| ctm.__init__ | dev_mode | 可配置 | ctm: params.model.dev_mode |
| ctm.__init__ | word_embedding_dim | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| ctm.__init__ | word_embeddings | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| ctm.__init__ | train_word_embeddings | 兼容占位 | 本模型不消费词嵌入冻结选项 |

## dtm.__init__

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| dtm.__init__ | vocab_size | 可配置 | dtm: params.vocab_size |
| dtm.__init__ | num_topics | 可配置 | dtm: params.num_topics |
| dtm.__init__ | time_slices | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| dtm.__init__ | doc_embedding_dim | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| dtm.__init__ | word_embedding_dim | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| dtm.__init__ | hidden_dim | 可配置 | dtm: params.hidden_dim |
| dtm.__init__ | encoder_dropout | 可配置 | dtm: params.model.encoder_dropout |
| dtm.__init__ | word_embeddings | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| dtm.__init__ | train_word_embeddings | 可配置 | dtm: params.model.train_word_embeddings |
| dtm.__init__ | kl_weight | 可配置 | dtm: params.model.kl_weight |
| dtm.__init__ | evolution_weight | 可配置 | dtm: params.model.evolution_weight |
| dtm.__init__ | dev_mode | 可配置 | dtm: params.model.dev_mode |

## nvdm.__init__

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| nvdm.__init__ | vocab_size | 可配置 | nvdm: params.vocab_size |
| nvdm.__init__ | num_topics | 可配置 | nvdm: params.num_topics |
| nvdm.__init__ | hidden_dim | 可配置 | nvdm: params.hidden_dim |
| nvdm.__init__ | dropout | 可配置 | nvdm: params.model.dropout |

## gsm.__init__

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| gsm.__init__ | vocab_size | 可配置 | gsm: params.vocab_size |
| gsm.__init__ | num_topics | 可配置 | gsm: params.num_topics |
| gsm.__init__ | hidden_dim | 可配置 | gsm: params.hidden_dim |
| gsm.__init__ | dropout | 可配置 | gsm: params.model.dropout |

## prodlda.__init__

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| prodlda.__init__ | vocab_size | 可配置 | prodlda: params.vocab_size |
| prodlda.__init__ | num_topics | 可配置 | prodlda: params.num_topics |
| prodlda.__init__ | hidden_dim | 可配置 | prodlda: params.hidden_dim |
| prodlda.__init__ | dropout | 可配置 | prodlda: params.model.dropout |
| prodlda.__init__ | variance | 可配置 | prodlda: params.model.variance |

## bertopic.__init__

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| bertopic.__init__ | vocab_size | 可配置 | bertopic: params.vocab_size |
| bertopic.__init__ | num_topics | 可配置 | bertopic: params.num_topics |
| bertopic.__init__ | embedding_model | 映射 | params.embedding.model_path；使用批准的本地权重，预处理与训练保持一致 |
| bertopic.__init__ | n_neighbors | 可配置 | bertopic: params.n_neighbors |
| bertopic.__init__ | n_components | 可配置 | bertopic: params.n_components |
| bertopic.__init__ | min_cluster_size | 可配置 | bertopic: params.min_cluster_size |
| bertopic.__init__ | min_samples | 可配置 | bertopic: params.min_samples |
| bertopic.__init__ | top_n_words | 可配置 | bertopic: params.top_n_words |
| bertopic.__init__ | language | 可配置 | bertopic: params.language |
| bertopic.__init__ | calculate_probabilities | 可配置 | bertopic: params.model.calculate_probabilities |
| bertopic.__init__ | verbose | 可配置 | bertopic: params.model.verbose |
| bertopic.__init__ | random_state | 可配置 | bertopic: params.random_state |

## bertopic.fit

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| bertopic.fit | texts | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| bertopic.fit | embeddings | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |

## theta.__init__

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| theta.__init__ | vocab_size | 可配置 | theta: params.vocab_size |
| theta.__init__ | num_topics | 可配置 | theta: params.num_topics |
| theta.__init__ | doc_embedding_dim | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| theta.__init__ | word_embedding_dim | 可配置 | theta: params.config.word_embedding_dim |
| theta.__init__ | hidden_dim | 可配置 | theta: params.hidden_dim |
| theta.__init__ | encoder_dropout | 可配置 | theta: params.model.encoder_dropout; theta: params.config.encoder_dropout |
| theta.__init__ | encoder_activation | 可配置 | theta: params.model.encoder_activation; theta: params.config.encoder_activation |
| theta.__init__ | word_embeddings | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| theta.__init__ | train_word_embeddings | 可配置 | theta: params.model.train_word_embeddings; theta: params.config.train_word_embeddings |
| theta.__init__ | kl_weight | 可配置 | theta: params.model.kl_weight |
| theta.__init__ | num_classes | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| theta.__init__ | contrastive_weight | 可配置 | theta: params.model.contrastive_weight |
| theta.__init__ | contrastive_temp | 可配置 | theta: params.model.contrastive_temp; theta: params.config.contrastive_temp |
| theta.__init__ | dev_mode | 可配置 | theta: params.model.dev_mode |

## BaselineTrainer.train_lda

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| BaselineTrainer.train_lda | max_iter | 可配置 | lda: params.max_iter |
| BaselineTrainer.train_lda | learning_method | 可配置 | lda: params.trainer.learning_method |

## BaselineTrainer.train_hdp

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| BaselineTrainer.train_hdp | max_topics | 可配置 | hdp: params.max_topics |
| BaselineTrainer.train_hdp | alpha | 可配置 | hdp: params.alpha |
| BaselineTrainer.train_hdp | gamma | 可配置 | hdp: params.trainer.gamma |

## BaselineTrainer.train_stm

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| BaselineTrainer.train_stm | max_iter | 可配置 | stm: params.max_iter |
| BaselineTrainer.train_stm | covariates | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| BaselineTrainer.train_stm | covariate_names | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |

## BaselineTrainer.train_btm

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| BaselineTrainer.train_btm | n_iter | 可配置 | btm: params.n_iter |
| BaselineTrainer.train_btm | alpha | 可配置 | btm: params.alpha |
| BaselineTrainer.train_btm | beta | 可配置 | btm: params.beta |

## BaselineTrainer.train_etm

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| BaselineTrainer.train_etm | embedding_dim | 可配置 | etm: params.embedding_dim |
| BaselineTrainer.train_etm | hidden_dim | 可配置 | etm: params.hidden_dim |
| BaselineTrainer.train_etm | dropout | 可配置 | etm: params.dropout |
| BaselineTrainer.train_etm | train_embeddings | 可配置 | etm: params.trainer.train_embeddings |
| BaselineTrainer.train_etm | use_pretrained_embeddings | 可配置 | etm: params.trainer.use_pretrained_embeddings |
| BaselineTrainer.train_etm | epochs | 可配置 | etm: params.epochs |
| BaselineTrainer.train_etm | batch_size | 可配置 | etm: params.batch_size |
| BaselineTrainer.train_etm | learning_rate | 可配置 | etm: params.learning_rate |
| BaselineTrainer.train_etm | early_stopping_patience | 映射 | params.patience |

## BaselineTrainer.train_ctm

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| BaselineTrainer.train_ctm | inference_type | 可配置 | ctm: params.inference_type |
| BaselineTrainer.train_ctm | model_type | 可配置 | ctm: params.trainer.model_type |
| BaselineTrainer.train_ctm | hidden_sizes | 可配置 | ctm: params.trainer.hidden_sizes |
| BaselineTrainer.train_ctm | epochs | 可配置 | ctm: params.epochs |
| BaselineTrainer.train_ctm | batch_size | 可配置 | ctm: params.batch_size |
| BaselineTrainer.train_ctm | learning_rate | 可配置 | ctm: params.learning_rate |
| BaselineTrainer.train_ctm | early_stopping_patience | 映射 | params.patience |

## BaselineTrainer.train_dtm

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| BaselineTrainer.train_dtm | epochs | 可配置 | dtm: params.epochs |
| BaselineTrainer.train_dtm | batch_size | 可配置 | dtm: params.batch_size |
| BaselineTrainer.train_dtm | learning_rate | 可配置 | dtm: params.learning_rate |
| BaselineTrainer.train_dtm | hidden_dim | 可配置 | dtm: params.hidden_dim |
| BaselineTrainer.train_dtm | embedding_dim | 可配置 | dtm: params.embedding_dim |

## BaselineTrainer.train_nvdm

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| BaselineTrainer.train_nvdm | epochs | 可配置 | nvdm: params.epochs |
| BaselineTrainer.train_nvdm | batch_size | 可配置 | nvdm: params.batch_size |

## BaselineTrainer.train_gsm

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| BaselineTrainer.train_gsm | epochs | 可配置 | gsm: params.epochs |
| BaselineTrainer.train_gsm | batch_size | 可配置 | gsm: params.batch_size |

## BaselineTrainer.train_prodlda

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| BaselineTrainer.train_prodlda | epochs | 可配置 | prodlda: params.epochs |
| BaselineTrainer.train_prodlda | batch_size | 可配置 | prodlda: params.batch_size |

## BaselineTrainer.train_bertopic

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| BaselineTrainer.train_bertopic | n_neighbors | 可配置 | bertopic: params.n_neighbors |
| BaselineTrainer.train_bertopic | n_components | 可配置 | bertopic: params.n_components |
| BaselineTrainer.train_bertopic | min_cluster_size | 可配置 | bertopic: params.min_cluster_size |
| BaselineTrainer.train_bertopic | min_samples | 可配置 | bertopic: params.min_samples |
| BaselineTrainer.train_bertopic | top_n_words | 可配置 | bertopic: params.top_n_words |
| BaselineTrainer.train_bertopic | language | 可配置 | bertopic: params.language |
| BaselineTrainer.train_bertopic | random_state | 可配置 | bertopic: params.random_state |

## BaselineTrainer.__init__

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| BaselineTrainer.__init__ | dataset | 宿主管理 | host dataset binding |
| BaselineTrainer.__init__ | num_topics | 可配置 | lda: params.num_topics; stm: params.num_topics; btm: params.num_topics; etm: params.num_topics; ctm: params.num_topics; dtm: params.num_topics; nvdm: params.num_topics; gsm: params.num_topics; prodlda: params.num_topics; bertopic: params.num_topics; theta: params.num_topics |
| BaselineTrainer.__init__ | vocab_size | 可配置 | lda: params.vocab_size; hdp: params.vocab_size; stm: params.vocab_size; btm: params.vocab_size; etm: params.vocab_size; ctm: params.vocab_size; dtm: params.vocab_size; nvdm: params.vocab_size; gsm: params.vocab_size; prodlda: params.vocab_size; bertopic: params.vocab_size; theta: params.vocab_size |
| BaselineTrainer.__init__ | user_id | 宿主管理 | host isolation |
| BaselineTrainer.__init__ | workspace_dir | 宿主管理 | job workspace |
| BaselineTrainer.__init__ | result_dir | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| BaselineTrainer.__init__ | data_dir | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| BaselineTrainer.__init__ | data_exp_dir | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| BaselineTrainer.__init__ | output_dir | 宿主管理 | job workspace |
| BaselineTrainer.__init__ | device | 宿主管理 | plan.device / 按独立模型研究及确认执行 |

## BaselineTrainer._train_neural_topic_model

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| BaselineTrainer._train_neural_topic_model | model_class | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| BaselineTrainer._train_neural_topic_model | model_name | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| BaselineTrainer._train_neural_topic_model | epochs | 可配置 | etm: params.epochs; ctm: params.epochs; dtm: params.epochs; nvdm: params.epochs; gsm: params.epochs; prodlda: params.epochs; theta: params.epochs |
| BaselineTrainer._train_neural_topic_model | batch_size | 可配置 | etm: params.batch_size; ctm: params.batch_size; dtm: params.batch_size; nvdm: params.batch_size; gsm: params.batch_size; prodlda: params.batch_size; theta: params.batch_size |
| BaselineTrainer._train_neural_topic_model | learning_rate | 可配置 | etm: params.learning_rate; ctm: params.learning_rate; dtm: params.learning_rate; nvdm: params.learning_rate; gsm: params.learning_rate; prodlda: params.learning_rate; theta: params.learning_rate |
| BaselineTrainer._train_neural_topic_model | hidden_dim | 可配置 | etm: params.hidden_dim; ctm: params.hidden_dim; dtm: params.hidden_dim; nvdm: params.hidden_dim; gsm: params.hidden_dim; prodlda: params.hidden_dim; theta: params.hidden_dim |

## BaselineTrainer.train_all

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| BaselineTrainer.train_all | models | 宿主管理 | plan.device / 按独立模型研究及确认执行 |

## train_word2vec_embeddings

| 来源/入口 | 参数 | 状态 | 实际传递/原因 |
|---|---|---|---|
| train_word2vec_embeddings | texts | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| train_word2vec_embeddings | vocab | 数据/任务绑定 | 由已批准的数据、矩阵维度、入口或隔离任务目录提供，不能用任意路径/张量绕过绑定 |
| train_word2vec_embeddings | embedding_dim | 可配置 | etm: params.embedding_dim |
| train_word2vec_embeddings | window | 可配置 | etm: params.word2vec.window |
| train_word2vec_embeddings | min_count | 可配置 | etm: params.word2vec.min_count |
| train_word2vec_embeddings | workers | 可配置 | etm: params.word2vec.workers |

