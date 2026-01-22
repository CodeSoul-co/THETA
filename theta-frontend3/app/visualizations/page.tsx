'use client';

import { useState, useEffect } from 'react';
import { Loader2, PieChart, BarChart3, Grid3X3, FileText, Sparkles, RefreshCw, AlertCircle } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { WorkspaceLayout } from '@/components/workspace-layout';
import { ETMAgentAPI, ResultInfo } from '@/lib/api/etm-agent';
import { 
  WordCloud, 
  TopicDistributionChart, 
  TopicHeatmap, 
  DocTopicChart, 
  TopicBubbleChart 
} from '@/components/visualizations';

type VisualizationData = {
  topicDistribution?: {
    topics: string[];
    proportions: number[];
    topic_words: Record<string, string[]>;
  };
  docTopicDistribution?: {
    documents: string[];
    distributions: number[][];
    num_topics: number;
  };
  topicSimilarity?: {
    topics: string[];
    similarity_matrix: number[][];
    topic_words: Record<string, string[]>;
  };
};

function VisualizationsContent() {
  const [results, setResults] = useState<ResultInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dataLoading, setDataLoading] = useState(false);
  const [selectedResult, setSelectedResult] = useState<ResultInfo | null>(null);
  const [visualizationData, setVisualizationData] = useState<VisualizationData>({});
  const [activeTab, setActiveTab] = useState('distribution');

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
        await loadVisualizationData(data[0].dataset, data[0].mode);
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

  const loadVisualizationData = async (dataset: string, mode: string) => {
    setDataLoading(true);
    try {
      // Load all visualization data in parallel
      const [topicDist, docTopicDist, topicSim] = await Promise.all([
        ETMAgentAPI.getVisualizationData(dataset, mode, 'topic_distribution').catch(() => null),
        ETMAgentAPI.getVisualizationData(dataset, mode, 'doc_topic_distribution').catch(() => null),
        ETMAgentAPI.getVisualizationData(dataset, mode, 'topic_similarity').catch(() => null),
      ]);

      setVisualizationData({
        topicDistribution: topicDist as VisualizationData['topicDistribution'],
        docTopicDistribution: docTopicDist as VisualizationData['docTopicDistribution'],
        topicSimilarity: topicSim as VisualizationData['topicSimilarity'],
      });
    } catch (error) {
      console.error('Failed to load visualization data:', error);
      setVisualizationData({});
    } finally {
      setDataLoading(false);
    }
  };

  const handleSelectResult = async (result: ResultInfo) => {
    setSelectedResult(result);
    await loadVisualizationData(result.dataset, result.mode);
  };

  // Prepare word cloud data from topic words
  const getWordCloudData = () => {
    const topicWords = visualizationData.topicDistribution?.topic_words || {};
    const proportions = visualizationData.topicDistribution?.proportions || [];
    
    const wordMap = new Map<string, number>();
    
    Object.entries(topicWords).forEach(([topicIdx, words]) => {
      const topicWeight = proportions[Number(topicIdx)] || 0.1;
      words.forEach((word, wordIdx) => {
        const weight = topicWeight * (1 - wordIdx * 0.05); // Decay by position
        const existing = wordMap.get(word) || 0;
        wordMap.set(word, Math.max(existing, weight));
      });
    });

    return Array.from(wordMap.entries())
      .map(([text, weight]) => ({ text, weight }))
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 50);
  };

  // Prepare topic distribution chart data
  const getTopicDistributionData = () => {
    const data = visualizationData.topicDistribution;
    if (!data) return [];

    return data.topics.map((name, idx) => ({
      name,
      value: data.proportions[idx],
      words: data.topic_words[String(idx)] || [],
    }));
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
        <h2 className="text-2xl font-semibold text-slate-900 mb-2">可视化图表</h2>
        <p className="text-slate-600">查看训练结果的可视化展示，支持交互式探索和导出</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* 结果选择器 */}
        <div className="lg:col-span-1">
          <Card className="p-4 bg-white border border-slate-200 sticky top-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-medium text-slate-900">选择数据集</h3>
              <Button variant="ghost" size="sm" onClick={loadResults}>
                <RefreshCw className="w-4 h-4" />
              </Button>
            </div>
            {results.length === 0 ? (
              <div className="text-center text-slate-500 text-sm py-8">
                <PieChart className="w-8 h-8 text-slate-300 mx-auto mb-2" />
                暂无结果，请先完成模型训练
              </div>
            ) : (
              <div className="space-y-2">
                {results.map((result) => (
                  <Card
                    key={`${result.dataset}-${result.mode}`}
                    className={`p-3 cursor-pointer transition-colors ${
                      selectedResult?.dataset === result.dataset &&
                      selectedResult?.mode === result.mode
                        ? 'bg-blue-50 border-blue-200'
                        : 'hover:bg-slate-50 border-slate-200'
                    }`}
                    onClick={() => handleSelectResult(result)}
                  >
                    <p className="font-medium text-slate-900 text-sm">
                      {result.dataset}
                    </p>
                    <p className="text-xs text-slate-500 mt-1">{result.mode}</p>
                  </Card>
                ))}
              </div>
            )}
          </Card>
        </div>

        {/* 可视化图表 */}
        <div className="lg:col-span-4">
          {selectedResult ? (
            dataLoading ? (
              <Card className="p-8 bg-white border border-slate-200 flex items-center justify-center">
                <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
                <span className="ml-3 text-slate-600">加载可视化数据...</span>
              </Card>
            ) : (
              <Tabs value={activeTab} onValueChange={setActiveTab}>
                <TabsList className="mb-6">
                  <TabsTrigger value="distribution" className="gap-2">
                    <BarChart3 className="w-4 h-4" />
                    主题分布
                  </TabsTrigger>
                  <TabsTrigger value="wordcloud" className="gap-2">
                    <Sparkles className="w-4 h-4" />
                    词云图
                  </TabsTrigger>
                  <TabsTrigger value="bubble" className="gap-2">
                    <PieChart className="w-4 h-4" />
                    气泡图
                  </TabsTrigger>
                  <TabsTrigger value="heatmap" className="gap-2">
                    <Grid3X3 className="w-4 h-4" />
                    热力图
                  </TabsTrigger>
                  <TabsTrigger value="doctopic" className="gap-2">
                    <FileText className="w-4 h-4" />
                    文档分布
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="distribution">
                  <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                    <TopicDistributionChart 
                      data={getTopicDistributionData()}
                      title="主题占比分布 (条形图)"
                      chartType="bar"
                    />
                    <TopicDistributionChart 
                      data={getTopicDistributionData()}
                      title="主题占比分布 (饼图)"
                      chartType="pie"
                    />
                  </div>
                </TabsContent>

                <TabsContent value="wordcloud">
                  <WordCloud 
                    words={getWordCloudData()}
                    title="主题词云图"
                  />
                </TabsContent>

                <TabsContent value="bubble">
                  <TopicBubbleChart 
                    topics={visualizationData.topicDistribution?.topics || []}
                    proportions={visualizationData.topicDistribution?.proportions || []}
                    topicWords={visualizationData.topicDistribution?.topic_words}
                    title="主题气泡图"
                  />
                </TabsContent>

                <TabsContent value="heatmap">
                  {visualizationData.topicSimilarity ? (
                    <TopicHeatmap 
                      matrix={visualizationData.topicSimilarity.similarity_matrix}
                      labels={visualizationData.topicSimilarity.topics}
                      topicWords={visualizationData.topicSimilarity.topic_words}
                      title="主题相似度热力图"
                    />
                  ) : (
                    <Card className="p-8 bg-white border border-slate-200 text-center">
                      <Grid3X3 className="w-12 h-12 text-slate-300 mx-auto mb-4" />
                      <p className="text-slate-500">热力图数据不可用</p>
                    </Card>
                  )}
                </TabsContent>

                <TabsContent value="doctopic">
                  {visualizationData.docTopicDistribution ? (
                    <DocTopicChart 
                      documents={visualizationData.docTopicDistribution.documents}
                      distributions={visualizationData.docTopicDistribution.distributions}
                      numTopics={visualizationData.docTopicDistribution.num_topics}
                      title="文档-主题分布"
                    />
                  ) : (
                    <Card className="p-8 bg-white border border-slate-200 text-center">
                      <FileText className="w-12 h-12 text-slate-300 mx-auto mb-4" />
                      <p className="text-slate-500">文档分布数据不可用</p>
                    </Card>
                  )}
                </TabsContent>
              </Tabs>
            )
          ) : (
            <Card className="p-8 bg-white border border-slate-200 text-center">
              <PieChart className="w-12 h-12 text-slate-300 mx-auto mb-4" />
              <p className="text-slate-500">选择一个数据集查看可视化图表</p>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

export default function VisualizationsPage() {
  return (
    <WorkspaceLayout currentStep="visualizations">
      <VisualizationsContent />
    </WorkspaceLayout>
  );
}
