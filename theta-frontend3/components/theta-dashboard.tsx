'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload,
  FileUp,
  Settings,
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  Zap,
  BarChart3,
  TrendingUp,
  ChevronRight,
  Play,
  Sparkles,
  FileText,
  Eye,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from '@/components/ui/sheet';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { ETMAgentAPI, TaskResponse } from '@/lib/api/etm-agent';
import { useETMWebSocket } from '@/hooks/use-etm-websocket';

// ============================================
// Types
// ============================================
interface QuickStartConfig {
  mode: 'zero_shot' | 'unsupervised' | 'supervised';
  num_topics: number;
  epochs: number;
  batch_size: number;
}

interface DashboardTask extends TaskResponse {
  isNew?: boolean;
}

// ============================================
// QuickAnalyzer Component - 拖拽上传区
// ============================================
interface QuickAnalyzerProps {
  onTaskCreated: (taskId: string) => void;
  config: QuickStartConfig;
  onConfigChange: (config: QuickStartConfig) => void;
}

function QuickAnalyzer({ onTaskCreated, config, onConfigChange }: QuickAnalyzerProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [showSettings, setShowSettings] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      await handleFileUpload(files[0]);
    }
  }, [config]);

  const handleFileUpload = async (file: File) => {
    setIsUploading(true);
    setUploadProgress(0);

    try {
      const result = await ETMAgentAPI.uploadAndAnalyze(
        file,
        config,
        (progress) => setUploadProgress(progress)
      );
      
      onTaskCreated(result.task_id);
    } catch (error) {
      console.error('Upload failed:', error);
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  return (
    <div className="relative">
      {/* 主上传区域 */}
      <motion.div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isUploading && fileInputRef.current?.click()}
        className={`
          relative overflow-hidden rounded-2xl border-2 border-dashed cursor-pointer
          transition-all duration-300 ease-out
          ${isDragging 
            ? 'border-blue-500 bg-blue-50/50 scale-[1.02]' 
            : 'border-slate-200 bg-gradient-to-br from-slate-50 to-white hover:border-blue-300 hover:bg-blue-50/30'
          }
          ${isUploading ? 'pointer-events-none' : ''}
        `}
        animate={{
          scale: isDragging ? 1.02 : 1,
        }}
      >
        {/* 背景动画 */}
        <div className="absolute inset-0 overflow-hidden">
          <motion.div
            className="absolute -top-20 -right-20 w-40 h-40 bg-blue-100 rounded-full opacity-50"
            animate={{
              scale: isDragging ? 1.5 : 1,
              opacity: isDragging ? 0.7 : 0.3,
            }}
          />
          <motion.div
            className="absolute -bottom-20 -left-20 w-60 h-60 bg-purple-100 rounded-full opacity-30"
            animate={{
              scale: isDragging ? 1.3 : 1,
            }}
          />
        </div>

        <div className="relative p-8 md:p-12 text-center">
          <AnimatePresence mode="wait">
            {isUploading ? (
              <motion.div
                key="uploading"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-4"
              >
                <div className="w-16 h-16 mx-auto bg-blue-100 rounded-full flex items-center justify-center">
                  <Loader2 className="w-8 h-8 text-blue-600 animate-spin" />
                </div>
                <div>
                  <p className="text-lg font-medium text-slate-900">上传并分析中...</p>
                  <p className="text-sm text-slate-500 mt-1">文件正在处理，即将开始分析</p>
                </div>
                <div className="max-w-xs mx-auto">
                  <Progress value={uploadProgress} className="h-2" />
                  <p className="text-xs text-slate-400 mt-2">{uploadProgress}%</p>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="idle"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-4"
              >
                <motion.div 
                  className="w-20 h-20 mx-auto bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center shadow-lg"
                  animate={{
                    y: isDragging ? -8 : 0,
                    rotate: isDragging ? 5 : 0,
                  }}
                >
                  <Upload className="w-10 h-10 text-white" />
                </motion.div>
                <div>
                  <p className="text-xl font-semibold text-slate-900">
                    拖拽文件到这里，立即开始分析
                  </p>
                  <p className="text-slate-500 mt-2">
                    支持 CSV, JSON, TXT 格式 · 无需任何额外配置
                  </p>
                </div>
                <div className="flex items-center justify-center gap-2 text-sm text-slate-400">
                  <Zap className="w-4 h-4" />
                  <span>Drag, Drop, Done.</span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.json,.txt,.xlsx"
          onChange={handleFileSelect}
          className="hidden"
        />
      </motion.div>

      {/* 设置按钮 */}
      <Popover open={showSettings} onOpenChange={setShowSettings}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            size="sm"
            className="absolute top-3 right-3 gap-2"
          >
            <Settings className="w-4 h-4" />
            <span className="hidden sm:inline">设置</span>
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-72" align="end">
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>分析模式</Label>
              <Select
                value={config.mode}
                onValueChange={(value: QuickStartConfig['mode']) => 
                  onConfigChange({ ...config, mode: value })
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="zero_shot">Zero-shot</SelectItem>
                  <SelectItem value="unsupervised">无监督</SelectItem>
                  <SelectItem value="supervised">有监督</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>主题数量: {config.num_topics}</Label>
              <Input
                type="range"
                min={5}
                max={50}
                value={config.num_topics}
                onChange={(e) => 
                  onConfigChange({ ...config, num_topics: parseInt(e.target.value) })
                }
              />
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-1">
                <Label className="text-xs">训练轮数</Label>
                <Input
                  type="number"
                  value={config.epochs}
                  onChange={(e) => 
                    onConfigChange({ ...config, epochs: parseInt(e.target.value) || 50 })
                  }
                  min={10}
                  max={200}
                  className="h-8"
                />
              </div>
              <div className="space-y-1">
                <Label className="text-xs">批大小</Label>
                <Input
                  type="number"
                  value={config.batch_size}
                  onChange={(e) => 
                    onConfigChange({ ...config, batch_size: parseInt(e.target.value) || 64 })
                  }
                  min={16}
                  max={256}
                  className="h-8"
                />
              </div>
            </div>
          </div>
        </PopoverContent>
      </Popover>
    </div>
  );
}

// ============================================
// LiveTaskFeed Component - 实时任务列表
// ============================================
interface LiveTaskFeedProps {
  tasks: DashboardTask[];
  onTaskClick: (task: DashboardTask) => void;
  selectedTaskId?: string;
}

function LiveTaskFeed({ tasks, onTaskClick, selectedTaskId }: LiveTaskFeedProps) {
  const getStatusConfig = (status: string) => {
    const configs: Record<string, { icon: React.ReactNode; color: string; bg: string }> = {
      queued: { icon: <Clock className="w-4 h-4" />, color: 'text-amber-600', bg: 'bg-amber-500' },
      pending: { icon: <Clock className="w-4 h-4" />, color: 'text-amber-600', bg: 'bg-amber-500' },
      running: { icon: <Loader2 className="w-4 h-4 animate-spin" />, color: 'text-blue-600', bg: 'bg-blue-500' },
      completed: { icon: <CheckCircle2 className="w-4 h-4" />, color: 'text-green-600', bg: 'bg-green-500' },
      failed: { icon: <XCircle className="w-4 h-4" />, color: 'text-red-600', bg: 'bg-red-500' },
    };
    return configs[status] || configs.pending;
  };

  const getStatusLabel = (status: string) => {
    const labels: Record<string, string> = {
      queued: '排队中',
      pending: '等待中',
      running: '运行中',
      completed: '已完成',
      failed: '失败',
    };
    return labels[status] || status;
  };

  if (tasks.length === 0) {
    return (
      <Card className="p-8 text-center border-dashed">
        <div className="w-16 h-16 mx-auto bg-slate-100 rounded-full flex items-center justify-center mb-4">
          <BarChart3 className="w-8 h-8 text-slate-400" />
        </div>
        <p className="text-slate-500">暂无任务</p>
        <p className="text-sm text-slate-400 mt-1">上传文件后，分析任务将在这里显示</p>
      </Card>
    );
  }

  return (
    <div className="space-y-3">
      <AnimatePresence mode="popLayout">
        {tasks.map((task, index) => {
          const statusConfig = getStatusConfig(task.status);
          const isSelected = selectedTaskId === task.task_id;
          
          return (
            <motion.div
              key={task.task_id}
              layout
              initial={{ opacity: 0, y: -20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ delay: index * 0.05 }}
              onClick={() => onTaskClick(task)}
              className={`
                relative overflow-hidden rounded-xl border cursor-pointer
                transition-all duration-200
                ${isSelected 
                  ? 'border-blue-500 ring-2 ring-blue-500/20 bg-blue-50/50' 
                  : 'border-slate-200 bg-white hover:border-slate-300 hover:shadow-md'
                }
              `}
            >
              {/* 进度条背景 */}
              <motion.div
                className={`absolute inset-0 ${statusConfig.bg} opacity-[0.08]`}
                initial={{ width: '0%' }}
                animate={{ width: `${task.progress}%` }}
                transition={{ duration: 0.5, ease: 'easeOut' }}
              />

              <div className="relative p-4">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className={statusConfig.color}>{statusConfig.icon}</span>
                      <span className="font-medium text-slate-900 truncate">
                        {task.dataset || '快速分析'}
                      </span>
                      {task.isNew && (
                        <Badge variant="secondary" className="bg-blue-100 text-blue-700 text-xs">
                          新
                        </Badge>
                      )}
                    </div>
                    <div className="flex items-center gap-3 text-xs text-slate-500">
                      <span>{task.mode || 'zero_shot'}</span>
                      <span>·</span>
                      <span>{task.num_topics || 20} 主题</span>
                      {task.current_step && task.status === 'running' && (
                        <>
                          <span>·</span>
                          <span className="text-blue-600">{task.current_step}</span>
                        </>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-3">
                    <div className="text-right">
                      <Badge 
                        variant="outline" 
                        className={`${statusConfig.color} border-current`}
                      >
                        {getStatusLabel(task.status)}
                      </Badge>
                      <p className="text-xs text-slate-400 mt-1">{task.progress}%</p>
                    </div>
                    <ChevronRight className="w-5 h-5 text-slate-300" />
                  </div>
                </div>

                {/* 进度条 */}
                <div className="mt-3">
                  <Progress value={task.progress} className="h-1.5" />
                </div>
              </div>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}

// ============================================
// ResultSheet Component - 结果抽屉
// ============================================
interface ResultSheetProps {
  task: DashboardTask | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

function ResultSheet({ task, open, onOpenChange }: ResultSheetProps) {
  if (!task) return null;

  const isCompleted = task.status === 'completed';
  const metrics = task.metrics || {};
  const topicWords = task.topic_words || {};

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full sm:max-w-xl overflow-y-auto">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-blue-600" />
            分析结果
          </SheetTitle>
          <SheetDescription>
            {task.dataset || '快速分析'} - {task.mode || 'zero_shot'}
          </SheetDescription>
        </SheetHeader>

        <div className="mt-6 space-y-6">
          {/* 状态概览 */}
          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                {isCompleted ? (
                  <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center">
                    <CheckCircle2 className="w-5 h-5 text-green-600" />
                  </div>
                ) : task.status === 'running' ? (
                  <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                    <Loader2 className="w-5 h-5 text-blue-600 animate-spin" />
                  </div>
                ) : (
                  <div className="w-10 h-10 bg-amber-100 rounded-full flex items-center justify-center">
                    <Clock className="w-5 h-5 text-amber-600" />
                  </div>
                )}
                <div>
                  <p className="font-medium text-slate-900">
                    {isCompleted ? '分析完成' : task.status === 'running' ? '分析中...' : '等待中'}
                  </p>
                  <p className="text-sm text-slate-500">{task.message || task.current_step}</p>
                </div>
              </div>
              <span className="text-2xl font-bold text-blue-600">{task.progress}%</span>
            </div>
            <Progress value={task.progress} className="mt-4 h-2" />
          </Card>

          {/* 评估指标 */}
          {isCompleted && Object.keys(metrics).length > 0 && (
            <div>
              <h4 className="font-medium text-slate-900 mb-3 flex items-center gap-2">
                <TrendingUp className="w-4 h-4" />
                评估指标
              </h4>
              <div className="grid grid-cols-2 gap-3">
                {Object.entries(metrics).map(([key, value]) => (
                  <Card key={key} className="p-3">
                    <p className="text-xs text-slate-500 uppercase tracking-wide">{key}</p>
                    <p className="text-xl font-bold text-slate-900 mt-1">
                      {typeof value === 'number' ? value.toFixed(3) : value}
                    </p>
                  </Card>
                ))}
              </div>
            </div>
          )}

          {/* 主题词 */}
          {isCompleted && Object.keys(topicWords).length > 0 && (
            <div>
              <h4 className="font-medium text-slate-900 mb-3 flex items-center gap-2">
                <FileText className="w-4 h-4" />
                主题词
              </h4>
              <div className="space-y-3">
                {Object.entries(topicWords).slice(0, 5).map(([topic, words]) => (
                  <Card key={topic} className="p-3">
                    <p className="text-sm font-medium text-slate-700 mb-2">{topic}</p>
                    <div className="flex flex-wrap gap-1.5">
                      {(words as string[]).slice(0, 8).map((word, i) => (
                        <Badge key={i} variant="secondary" className="text-xs">
                          {word}
                        </Badge>
                      ))}
                    </div>
                  </Card>
                ))}
              </div>
            </div>
          )}

          {/* 查看完整结果按钮 */}
          {isCompleted && (
            <Button className="w-full gap-2" variant="default">
              <Eye className="w-4 h-4" />
              查看完整可视化
            </Button>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}

// ============================================
// ThetaDashboard Main Component
// ============================================
export function ThetaDashboard() {
  const [tasks, setTasks] = useState<DashboardTask[]>([]);
  const [selectedTask, setSelectedTask] = useState<DashboardTask | null>(null);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [config, setConfig] = useState<QuickStartConfig>({
    mode: 'zero_shot',
    num_topics: 20,
    epochs: 50,
    batch_size: 64,
  });
  
  const { lastMessage, subscribe } = useETMWebSocket();
  const sessionTaskIds = useRef<Set<string>>(new Set());

  // 加载任务列表
  useEffect(() => {
    loadTasks();
  }, []);

  // 轮询更新
  useEffect(() => {
    const interval = setInterval(() => {
      const activeTasks = tasks.filter(t => 
        t.status === 'pending' || t.status === 'running' || t.status === 'queued'
      );
      if (activeTasks.length > 0) {
        loadTasks();
      }
    }, 3000);
    return () => clearInterval(interval);
  }, [tasks]);

  // WebSocket 更新
  useEffect(() => {
    if (lastMessage?.task_id) {
      setTasks(prev => prev.map(t => {
        if (t.task_id === lastMessage.task_id) {
          const updated = {
            ...t,
            status: lastMessage.status as TaskResponse['status'] || t.status,
            current_step: lastMessage.step || t.current_step,
            progress: lastMessage.progress ?? t.progress,
            message: lastMessage.message || t.message,
          };
          
          // 自动打开已完成的会话任务
          if (updated.status === 'completed' && sessionTaskIds.current.has(t.task_id)) {
            setSelectedTask(updated);
            setSheetOpen(true);
          }
          
          return updated;
        }
        return t;
      }));
    }
  }, [lastMessage]);

  const loadTasks = async () => {
    try {
      const data = await ETMAgentAPI.getTasks();
      setTasks(prev => {
        // 保持 isNew 标记
        return data.map(task => ({
          ...task,
          isNew: prev.find(p => p.task_id === task.task_id)?.isNew || false,
        }));
      });
    } catch (error) {
      console.error('Failed to load tasks:', error);
    }
  };

  const handleTaskCreated = (taskId: string) => {
    sessionTaskIds.current.add(taskId);
    subscribe(taskId);
    
    // 添加新任务到列表顶部
    const newTask: DashboardTask = {
      task_id: taskId,
      status: 'queued',
      progress: 0,
      isNew: true,
    };
    setTasks(prev => [newTask, ...prev]);
    
    // 加载最新数据
    setTimeout(loadTasks, 500);
  };

  const handleTaskClick = (task: DashboardTask) => {
    setSelectedTask(task);
    setSheetOpen(true);
    
    // 清除 isNew 标记
    setTasks(prev => prev.map(t => 
      t.task_id === task.task_id ? { ...t, isNew: false } : t
    ));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50/30">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12">
        {/* Header */}
        <motion.div 
          className="text-center mb-8 sm:mb-12"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h1 className="text-3xl sm:text-4xl font-bold text-slate-900 tracking-tight">
            THETA Dashboard
          </h1>
          <p className="text-slate-500 mt-2 text-lg">
            主题模型分析平台 · 拖拽即分析
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* 左侧：上传区域 */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
          >
            <h2 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
              <Zap className="w-5 h-5 text-amber-500" />
              快速分析
            </h2>
            <QuickAnalyzer
              onTaskCreated={handleTaskCreated}
              config={config}
              onConfigChange={setConfig}
            />
          </motion.div>

          {/* 右侧：任务列表 */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-slate-900 flex items-center gap-2">
                <Play className="w-5 h-5 text-blue-500" />
                任务列表
              </h2>
              <Badge variant="outline" className="text-slate-500">
                {tasks.filter(t => t.status === 'running' || t.status === 'queued').length} 运行中
              </Badge>
            </div>
            <LiveTaskFeed
              tasks={tasks}
              onTaskClick={handleTaskClick}
              selectedTaskId={selectedTask?.task_id}
            />
          </motion.div>
        </div>

        {/* 结果抽屉 */}
        <ResultSheet
          task={selectedTask}
          open={sheetOpen}
          onOpenChange={setSheetOpen}
        />
      </div>
    </div>
  );
}

export default ThetaDashboard;
