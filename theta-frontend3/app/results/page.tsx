'use client';

import { useState, useEffect } from 'react';
import { BarChart3, Loader2, ExternalLink, Download, AlertCircle, RefreshCw } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { WorkspaceLayout } from '@/components/workspace-layout';
import { ETMAgentAPI, ResultInfo } from '@/lib/api/etm-agent';

// 格式化时间戳 (格式: "20240115_123456")
function formatTimestamp(timestamp: string): string {
  if (!timestamp) return '未知时间';
  
  try {
    // 尝试解析格式 "YYYYMMDD_HHMMSS"
    if (timestamp.match(/^\d{8}_\d{6}$/)) {
      const year = timestamp.substring(0, 4);
      const month = timestamp.substring(4, 6);
      const day = timestamp.substring(6, 8);
      const hour = timestamp.substring(9, 11);
      const minute = timestamp.substring(11, 13);
      const second = timestamp.substring(13, 15);
      return `${year}-${month}-${day} ${hour}:${minute}:${second}`;
    }
    // 尝试作为 ISO 日期解析
    return new Date(timestamp).toLocaleString('zh-CN');
  } catch {
    return timestamp;
  }
}

function ResultsContent() {
  const [results, setResults] = useState<ResultInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedResult, setSelectedResult] = useState<ResultInfo | null>(null);
  const [topicWords, setTopicWords] = useState<Record<string, string[]> | null>(null);
  const [metrics, setMetrics] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    loadResults();
  }, []);

  const loadResults = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Add timeout to prevent infinite loading
      const timeoutPromise = new Promise<never>((_, reject) => {
        setTimeout(() => reject(new Error('请求超时，请检查网络连接或后端服务状态')), 10000);
      });
      
      const data = await Promise.race([
        ETMAgentAPI.getResults(),
        timeoutPromise
      ]);
      
      setResults(data || []);
      if (data && data.length > 0) {
        setSelectedResult(data[0]);
        await loadResultDetails(data[0].dataset, data[0].mode);
      }
    } catch (error) {
      console.error('Failed to load results:', error);
      const errorMessage = error instanceof Error ? error.message : '加载结果失败';
      setError(errorMessage);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const loadResultDetails = async (dataset: string, mode: string) => {
    try {
      const [words, metricsData] = await Promise.all([
        ETMAgentAPI.getTopicWords(dataset, mode),
        ETMAgentAPI.getMetrics(dataset, mode).catch(() => null),
      ]);
      setTopicWords(words);
      setMetrics(metricsData);
    } catch (error) {
      console.error('Failed to load result details:', error);
      setTopicWords(null);
      setMetrics(null);
    }
  };

  const handleSelectResult = async (result: ResultInfo) => {
    setSelectedResult(result);
    await loadResultDetails(result.dataset, result.mode);
  };

  const handleExportResults = (format: 'csv' | 'json') => {
    if (!selectedResult || !topicWords || !metrics) {
      return;
    }

    if (format === 'csv') {
      // 导出 CSV 格式
      let csvContent = '主题,主题词\n';
      Object.entries(topicWords).forEach(([topic, words]) => {
        csvContent += `主题${topic},"${words.join(',')}"\n`;
      });
      
      // 添加评估指标
      csvContent += '\n评估指标,数值\n';
      Object.entries(metrics).forEach(([key, value]) => {
        csvContent += `${key},${value}\n`;
      });

      // 下载文件
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const link = document.createElement('a');
      const url = URL.createObjectURL(blob);
      link.setAttribute('href', url);
      link.setAttribute('download', `${selectedResult.dataset}_${selectedResult.mode}_results.csv`);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } else {
      // 导出 JSON 格式
      const jsonData = {
        dataset: selectedResult.dataset,
        mode: selectedResult.mode,
        timestamp: selectedResult.timestamp,
        metrics: metrics,
        topic_words: topicWords,
      };

      const blob = new Blob([JSON.stringify(jsonData, null, 2)], { type: 'application/json' });
      const link = document.createElement('a');
      const url = URL.createObjectURL(blob);
      link.setAttribute('href', url);
      link.setAttribute('download', `${selectedResult.dataset}_${selectedResult.mode}_results.json`);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  const handleViewReport = () => {
    if (!selectedResult) {
      return;
    }
    // 跳转到可视化页面，并传递参数
    window.location.href = `/visualizations?dataset=${selectedResult.dataset}&mode=${selectedResult.mode}`;
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4">
        <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
        <p className="text-slate-600">加载结果中...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8 h-full overflow-auto">
        <div className="max-w-2xl mx-auto">
          <Card className="p-6 bg-red-50 border-red-200">
            <div className="flex items-start gap-4">
              <AlertCircle className="w-6 h-6 text-red-600 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <h3 className="font-semibold text-red-900 mb-2">加载失败</h3>
                <p className="text-red-700 text-sm mb-4 whitespace-pre-line">{error}</p>
                <Button onClick={loadResults} variant="outline" size="sm" className="gap-2">
                  <RefreshCw className="w-4 h-4" />
                  重试
                </Button>
              </div>
            </div>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="p-8 h-full overflow-auto">
      <div className="mb-6">
        <h2 className="text-2xl font-semibold text-slate-900 mb-2">分析结果</h2>
        <p className="text-slate-600">查看训练完成的主题模型分析结果</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 结果列表 */}
        <div className="lg:col-span-1 space-y-4">
          <Card className="p-4 bg-white border border-slate-200">
            <h3 className="font-medium text-slate-900 mb-4">结果列表</h3>
            {results.length === 0 ? (
              <div className="text-center text-slate-500 text-sm py-8">
                <BarChart3 className="w-8 h-8 text-slate-300 mx-auto mb-2" />
                <p>暂无结果</p>
                <p className="text-xs text-slate-400 mt-1">请先完成模型训练</p>
              </div>
            ) : (
              <div className="space-y-2 max-h-[600px] overflow-y-auto">
                {results.map((result) => {
                  const isSelected = selectedResult?.dataset === result.dataset &&
                    selectedResult?.mode === result.mode;
                  
                  return (
                    <Card
                      key={`${result.dataset}-${result.mode}`}
                      className={`p-3 cursor-pointer transition-all ${
                        isSelected
                          ? 'bg-blue-50 border-blue-300 shadow-sm'
                          : 'hover:bg-slate-50 border-slate-200 hover:shadow-sm'
                      }`}
                      onClick={() => handleSelectResult(result)}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1 min-w-0">
                          <p className="font-medium text-slate-900 text-sm truncate">
                            {result.dataset}
                          </p>
                          <div className="flex items-center gap-2 mt-1">
                            <span className="text-xs px-2 py-0.5 bg-slate-200 text-slate-700 rounded">
                              {result.mode}
                            </span>
                            {result.num_topics && (
                              <span className="text-xs text-slate-500">
                                {result.num_topics} 主题
                              </span>
                            )}
                          </div>
                          <p className="text-xs text-slate-400 mt-1">
                            {formatTimestamp(result.timestamp)}
                          </p>
                          {result.metrics && (
                            <div className="flex items-center gap-3 mt-2 text-xs">
                              {result.metrics.topic_coherence_avg !== undefined && (
                                <span className="text-slate-600">
                                  一致性: <span className="font-medium text-blue-600">
                                    {result.metrics.topic_coherence_avg.toFixed(3)}
                                  </span>
                                </span>
                              )}
                            </div>
                          )}
                        </div>
                        <BarChart3 className={`w-5 h-5 flex-shrink-0 ml-2 ${
                          isSelected ? 'text-blue-600' : 'text-slate-400'
                        }`} />
                      </div>
                    </Card>
                  );
                })}
              </div>
            )}
          </Card>
        </div>

        {/* 结果详情 */}
        <div className="lg:col-span-2 space-y-6">
          {selectedResult ? (
            <>
              {/* 结果基本信息 */}
              <Card className="p-6 bg-white border border-slate-200">
                <h3 className="font-medium text-slate-900 mb-4">结果信息</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <p className="text-xs text-slate-500 mb-1">数据集</p>
                    <p className="font-medium text-slate-900">{selectedResult.dataset}</p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-500 mb-1">训练模式</p>
                    <p className="font-medium text-slate-900">{selectedResult.mode}</p>
                  </div>
                  {selectedResult.num_topics && (
                    <div>
                      <p className="text-xs text-slate-500 mb-1">主题数</p>
                      <p className="font-medium text-slate-900">{selectedResult.num_topics}</p>
                    </div>
                  )}
                  {selectedResult.epochs_trained && (
                    <div>
                      <p className="text-xs text-slate-500 mb-1">训练轮数</p>
                      <p className="font-medium text-slate-900">{selectedResult.epochs_trained}</p>
                    </div>
                  )}
                </div>
              </Card>

              {/* 评估指标 */}
              {metrics && Object.keys(metrics).length > 0 && (
                <Card className="p-6 bg-white border border-slate-200">
                  <h3 className="font-medium text-slate-900 mb-4">评估指标</h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                    {Object.entries(metrics).map(([key, value]) => {
                      // 跳过非数字值或嵌套对象
                      if (typeof value !== 'number' || isNaN(value)) {
                        return null;
                      }
                      
                      // 格式化指标名称
                      const formatKey = (k: string): string => {
                        const keyMap: Record<string, string> = {
                          topic_coherence_avg: '主题一致性 (平均)',
                          topic_coherence_per_topic: '主题一致性 (每主题)',
                          topic_diversity_td: '主题多样性 (TD)',
                          topic_diversity_irbo: '主题多样性 (IRBO)',
                          best_val_loss: '最佳验证损失',
                          test_loss: '测试损失',
                          perplexity: '困惑度',
                        };
                        return keyMap[k] || k.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                      };
                      
                      return (
                        <div key={key} className="text-center p-4 bg-gradient-to-br from-blue-50 to-slate-50 rounded-lg border border-slate-200">
                          <p className="text-2xl font-bold text-blue-600 mb-1">
                            {value.toFixed(4)}
                          </p>
                          <p className="text-xs text-slate-600">{formatKey(key)}</p>
                        </div>
                      );
                    })}
                  </div>
                  {Object.keys(metrics).filter(k => typeof metrics[k] === 'number' && !isNaN(metrics[k] as number)).length === 0 && (
                    <p className="text-center text-slate-500 text-sm py-4">暂无可用指标</p>
                  )}
                </Card>
              )}

              {/* 主题词 */}
              {topicWords && Object.keys(topicWords).length > 0 ? (
                <Card className="p-6 bg-white border border-slate-200">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="font-medium text-slate-900">主题词分析</h3>
                    <p className="text-sm text-slate-500">
                      共 {Object.keys(topicWords).length} 个主题
                    </p>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {Object.entries(topicWords)
                      .sort(([a], [b]) => {
                        // 尝试按数字排序
                        const numA = parseInt(a);
                        const numB = parseInt(b);
                        if (!isNaN(numA) && !isNaN(numB)) {
                          return numA - numB;
                        }
                        return a.localeCompare(b);
                      })
                      .map(([topic, words]) => (
                        <div key={topic} className="p-4 bg-gradient-to-br from-slate-50 to-blue-50 rounded-lg border border-slate-200 hover:shadow-md transition-shadow">
                          <div className="flex items-center justify-between mb-3">
                            <p className="font-semibold text-slate-900">
                              主题 {topic.replace(/^topic_?/i, '').replace(/_/g, ' ')}
                            </p>
                            <span className="text-xs text-slate-500 bg-white px-2 py-1 rounded">
                              {words.length} 个词
                            </span>
                          </div>
                          <div className="flex flex-wrap gap-1.5">
                            {words.slice(0, 10).map((word, index) => (
                              <span
                                key={index}
                                className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${
                                  index < 3
                                    ? 'bg-blue-600 text-white'
                                    : index < 6
                                    ? 'bg-blue-100 text-blue-700'
                                    : 'bg-slate-100 text-slate-700'
                                }`}
                                title={word}
                              >
                                {word}
                              </span>
                            ))}
                            {words.length > 10 && (
                              <span className="px-2.5 py-1 rounded-md text-xs text-slate-500 bg-slate-100">
                                +{words.length - 10} 更多
                              </span>
                            )}
                          </div>
                        </div>
                      ))}
                  </div>
                </Card>
              ) : (
                selectedResult && (
                  <Card className="p-6 bg-white border border-slate-200">
                    <div className="text-center py-8">
                      <p className="text-slate-500">暂无主题词数据</p>
                      {!selectedResult.has_topic_words && (
                        <p className="text-xs text-slate-400 mt-2">
                          该结果可能尚未生成主题词文件
                        </p>
                      )}
                    </div>
                  </Card>
                )
              )}

              {/* 操作按钮 */}
              <div className="flex flex-wrap gap-3">
                <Button 
                  variant="outline" 
                  className="gap-2"
                  onClick={() => handleExportResults('csv')}
                  disabled={!topicWords || !metrics}
                >
                  <Download className="w-4 h-4" />
                  导出 CSV
                </Button>
                <Button 
                  variant="outline" 
                  className="gap-2"
                  onClick={() => handleExportResults('json')}
                  disabled={!topicWords || !metrics}
                >
                  <Download className="w-4 h-4" />
                  导出 JSON
                </Button>
                {selectedResult.has_visualizations && (
                  <Button 
                    variant="outline" 
                    className="gap-2"
                    onClick={() => handleViewReport()}
                  >
                    <ExternalLink className="w-4 h-4" />
                    查看详细报告
                  </Button>
                )}
              </div>
            </>
          ) : (
            <Card className="p-8 bg-white border border-slate-200 text-center">
              <BarChart3 className="w-12 h-12 text-slate-300 mx-auto mb-4" />
              <p className="text-slate-500">选择一个结果查看详情</p>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

export default function ResultsPage() {
  return (
    <WorkspaceLayout currentStep="results">
      <ResultsContent />
    </WorkspaceLayout>
  );
}
