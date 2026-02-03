'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { Send, Loader2, Zap, Play, X, Plus, Database, RefreshCw } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { WorkspaceLayout } from '@/components/workspace-layout';
import { TaskProgressTracker, TaskList } from '@/components/project/task-progress-tracker';
import { useETMWebSocket } from '@/hooks/use-etm-websocket';
import { ETMAgentAPI, TaskResponse, CreateTaskRequest, DatasetInfo } from '@/lib/api/etm-agent';

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  action?: string;
  data?: Record<string, unknown>;
}

function TrainingContent() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [tasks, setTasks] = useState<TaskResponse[]>([]);
  const [selectedTask, setSelectedTask] = useState<TaskResponse | null>(null);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // 新建任务表单状态
  const [newTaskForm, setNewTaskForm] = useState({
    dataset: '',
    mode: 'zero_shot' as 'zero_shot' | 'unsupervised' | 'supervised',
    numTopics: 20,
    epochs: 100,
    batchSize: 64,
  });

  const { lastMessage, sendMessage: wsSend, subscribe } = useETMWebSocket();

  useEffect(() => {
    loadTasks();
    loadDatasets();
  }, []);

  const loadDatasets = useCallback(async () => {
    try {
      const data = await ETMAgentAPI.getDatasets();
      setDatasets(data);
    } catch (error) {
      console.error('Failed to load datasets:', error);
    }
  }, []);

  useEffect(() => {
    if (lastMessage) {
      if (lastMessage.type === 'step_update') {
        const systemMsg: Message = {
          id: `sys-${Date.now()}`,
          role: 'system',
          content: `**${lastMessage.step}**: ${lastMessage.message}`,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, systemMsg]);

        if (lastMessage.task_id) {
          updateTaskStatus(lastMessage.task_id);
        }
      } else if (lastMessage.type === 'task_update') {
        updateTaskStatus(lastMessage.task_id as string);
      }
    }
  }, [lastMessage]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const loadTasks = async () => {
    try {
      const data = await ETMAgentAPI.getTasks();
      setTasks(data);
    } catch (error) {
      console.error('Failed to load tasks:', error);
      const errorMessage = error instanceof Error ? error.message : '未知错误';
      if (errorMessage.includes('无法连接')) {
        const systemMsg: Message = {
          id: `error-${Date.now()}`,
          role: 'system',
          content: `⚠️ ${errorMessage}\n\n请确保后端服务正在运行`,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, systemMsg]);
      }
    }
  };

  const updateTaskStatus = async (taskId: string) => {
    try {
      const task = await ETMAgentAPI.getTask(taskId);
      setTasks((prev) => prev.map((t) => (t.task_id === taskId ? task : t)));
      if (selectedTask?.task_id === taskId) {
        setSelectedTask(task);
      }
    } catch (error) {
      console.error('Failed to update task status:', error);
    }
  };

  // 创建新任务
  const handleCreateTask = async () => {
    if (!newTaskForm.dataset) return;

    setIsLoading(true);
    try {
      const task = await ETMAgentAPI.createTask({
        dataset: newTaskForm.dataset,
        mode: newTaskForm.mode,
        num_topics: newTaskForm.numTopics,
        epochs: newTaskForm.epochs,
        batch_size: newTaskForm.batchSize,
      });

      setTasks((prev) => [task, ...prev]);
      setSelectedTask(task);
      subscribe(task.task_id);
      setShowCreateDialog(false);

      const systemMsg: Message = {
        id: `sys-${Date.now()}`,
        role: 'system',
        content: `✅ 任务已创建: ${task.task_id}\n数据集: ${newTaskForm.dataset}\n模式: ${newTaskForm.mode}\n主题数: ${newTaskForm.numTopics}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, systemMsg]);
    } catch (error: any) {
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        role: 'system',
        content: `❌ 创建任务失败: ${error.message || '未知错误'}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: input,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await ETMAgentAPI.chat(input);

      if (response.action === 'start_task' && response.data) {
        const taskRequest = response.data as CreateTaskRequest;
        const task = await ETMAgentAPI.createTask(taskRequest);

        const assistantMessage: Message = {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: `任务已创建！任务 ID: ${task.task_id}\n状态: ${task.status}\n进度: ${task.progress}%`,
          timestamp: new Date(),
          action: 'task_created',
          data: { task_id: task.task_id },
        };

        setMessages((prev) => [...prev, assistantMessage]);
        setTasks((prev) => [task, ...prev]);
        setSelectedTask(task);
        subscribe(task.task_id);
      } else {
        const assistantMessage: Message = {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: response.message,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, assistantMessage]);
      }
    } catch (error: any) {
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: `错误: ${error.message || '处理请求失败'}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCancelTask = async (taskId: string) => {
    try {
      await ETMAgentAPI.cancelTask(taskId);
      await loadTasks();
      if (selectedTask?.task_id === taskId) {
        setSelectedTask(null);
      }
    } catch (error) {
      console.error('Failed to cancel task:', error);
    }
  };

  return (
    <>
      <div className="flex h-full">
        {/* 左侧主内容区 */}
        <div className="flex-1 flex flex-col p-8 overflow-auto">
          <div className="mb-6 flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-semibold text-slate-900 mb-2">模型训练</h2>
              <p className="text-slate-600">创建和管理 ETM 主题模型训练任务</p>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" onClick={loadTasks}>
                <RefreshCw className="w-4 h-4 mr-2" />
                刷新
              </Button>
              <Button onClick={() => {
                loadDatasets();
                setShowCreateDialog(true);
              }}>
                <Plus className="w-4 h-4 mr-2" />
                新建任务
              </Button>
            </div>
          </div>

          {/* 快捷操作 */}
          {messages.length === 0 && tasks.length === 0 && (
            <div className="mb-6">
              <Card className="p-6 bg-white border border-slate-200">
                <div className="flex items-start gap-4">
                  <div className="w-12 h-12 rounded-xl bg-blue-100 flex items-center justify-center flex-shrink-0">
                    <Zap className="w-6 h-6 text-blue-600" />
                  </div>
                  <div className="flex-1">
                    <h3 className="font-medium text-slate-900 mb-2">快速开始</h3>
                    <p className="text-sm text-slate-500 mb-4">选择一个示例命令开始训练，或点击"新建任务"按钮</p>
                    <div className="flex flex-wrap gap-2">
                      {[
                        '训练 socialTwitter 数据集',
                        '使用 zero_shot 模式训练，20 个主题',
                        '训练 hatespeech，supervised 模式',
                      ].map((example) => (
                        <button
                          key={example}
                          onClick={() => setInput(example)}
                          className="px-4 py-2 text-sm bg-slate-100 hover:bg-slate-200 rounded-lg text-slate-700 transition-colors"
                        >
                          {example}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </Card>
            </div>
          )}

          {/* 选中任务的详情 */}
          {selectedTask && (
            <div className="mb-6">
              <TaskProgressTracker
                taskId={selectedTask.task_id}
                onComplete={(task) => {
                  updateTaskStatus(task.task_id);
                }}
                onCancel={() => {
                  setSelectedTask(null);
                  loadTasks();
                }}
              />
            </div>
          )}

          {/* 任务列表 */}
          <div className="flex-1">
            <h3 className="font-medium text-slate-900 mb-4">
              任务列表 
              {tasks.length > 0 && <Badge variant="secondary" className="ml-2">{tasks.length}</Badge>}
            </h3>
            {tasks.length === 0 ? (
              <Card className="p-8 bg-white border border-slate-200 text-center">
                <Play className="w-12 h-12 text-slate-300 mx-auto mb-4" />
                <p className="text-slate-500 mb-2">暂无训练任务</p>
                <p className="text-sm text-slate-400 mb-4">点击"新建任务"按钮创建训练任务</p>
                <Button onClick={() => setShowCreateDialog(true)}>
                  <Plus className="w-4 h-4 mr-2" />
                  创建第一个任务
                </Button>
              </Card>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {tasks.map((task) => (
                  <Card
                    key={task.task_id}
                    className={`p-4 cursor-pointer transition-all hover:shadow-md ${
                      selectedTask?.task_id === task.task_id
                        ? 'bg-blue-50 border-blue-200 ring-2 ring-blue-100'
                        : 'bg-white border-slate-200'
                    }`}
                    onClick={() => setSelectedTask(task)}
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <Database className="w-4 h-4 text-slate-400" />
                          <p className="font-medium text-slate-900 truncate">
                            {task.dataset}
                          </p>
                        </div>
                        <p className="text-xs text-slate-500 mt-1">
                          模式: {task.mode || 'zero_shot'} · 主题数: {task.num_topics || 20}
                        </p>
                      </div>
                      {task.status !== 'completed' && task.status !== 'failed' && task.status !== 'cancelled' && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleCancelTask(task.task_id);
                          }}
                          className="h-6 w-6 p-0 hover:bg-red-100 hover:text-red-600"
                        >
                          <X className="w-4 h-4" />
                        </Button>
                      )}
                    </div>
                    {(task.status === 'running' || task.status === 'pending') && (
                      <Progress value={task.progress} className="h-2 mb-2" />
                    )}
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-slate-500">
                        {task.status === 'running' || task.status === 'pending' ? `${task.progress}%` : ''}
                      </span>
                      <Badge className={
                        task.status === 'completed' ? 'bg-green-100 text-green-700 hover:bg-green-100' :
                        task.status === 'failed' ? 'bg-red-100 text-red-700 hover:bg-red-100' :
                        task.status === 'running' ? 'bg-blue-100 text-blue-700 hover:bg-blue-100' :
                        task.status === 'cancelled' ? 'bg-slate-100 text-slate-700 hover:bg-slate-100' :
                        'bg-amber-100 text-amber-700 hover:bg-amber-100'
                      }>
                        {task.status === 'completed' ? '已完成' :
                         task.status === 'failed' ? '失败' :
                         task.status === 'running' ? '运行中' :
                         task.status === 'cancelled' ? '已取消' :
                         task.status === 'pending' ? '等待中' : task.status}
                      </Badge>
                    </div>
                    <p className="text-xs text-slate-400 mt-2">
                      {task.created_at ? new Date(task.created_at).toLocaleString('zh-CN') : ''}
                    </p>
                  </Card>
                ))}
              </div>
            )}
          </div>

          {/* 输入框 */}
          <div className="mt-6 pt-4 border-t border-slate-200">
            <form onSubmit={handleSubmit} className="flex gap-2">
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="输入训练命令，例如：训练 socialTwitter 数据集，20 个主题..."
                className="flex-1"
                disabled={isLoading}
              />
              <Button type="submit" disabled={isLoading || !input.trim()}>
                {isLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
              </Button>
            </form>
          </div>
        </div>
      </div>

      {/* 创建任务对话框 */}
      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent className="sm:max-w-md bg-white">
          <DialogHeader>
            <DialogTitle>新建训练任务</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label>数据集 <span className="text-red-500">*</span></Label>
              <Select
                value={newTaskForm.dataset}
                onValueChange={(v) => setNewTaskForm(prev => ({ ...prev, dataset: v }))}
              >
                <SelectTrigger>
                  <SelectValue placeholder="选择数据集" />
                </SelectTrigger>
                <SelectContent>
                  {datasets.map((ds) => (
                    <SelectItem key={ds.name} value={ds.name}>
                      <div className="flex items-center gap-2">
                        <Database className="h-4 w-4 text-slate-400" />
                        <span>{ds.name}</span>
                        {ds.size && (
                          <Badge variant="secondary" className="text-xs">
                            {ds.size} 条
                          </Badge>
                        )}
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>训练模式</Label>
              <Select
                value={newTaskForm.mode}
                onValueChange={(v) => setNewTaskForm(prev => ({ ...prev, mode: v as typeof newTaskForm.mode }))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="zero_shot">Zero-shot (直接嵌入)</SelectItem>
                  <SelectItem value="unsupervised">Unsupervised (无监督微调)</SelectItem>
                  <SelectItem value="supervised">Supervised (监督微调)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="grid grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label>主题数</Label>
                <Input
                  type="number"
                  min={5}
                  max={100}
                  value={newTaskForm.numTopics}
                  onChange={(e) => setNewTaskForm(prev => ({ ...prev, numTopics: parseInt(e.target.value) || 20 }))}
                />
              </div>
              <div className="space-y-2">
                <Label>训练轮数</Label>
                <Input
                  type="number"
                  min={10}
                  max={500}
                  value={newTaskForm.epochs}
                  onChange={(e) => setNewTaskForm(prev => ({ ...prev, epochs: parseInt(e.target.value) || 100 }))}
                />
              </div>
              <div className="space-y-2">
                <Label>批次大小</Label>
                <Input
                  type="number"
                  min={16}
                  max={256}
                  value={newTaskForm.batchSize}
                  onChange={(e) => setNewTaskForm(prev => ({ ...prev, batchSize: parseInt(e.target.value) || 64 }))}
                />
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowCreateDialog(false)}>
              取消
            </Button>
            <Button
              onClick={handleCreateTask}
              disabled={!newTaskForm.dataset || isLoading}
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Play className="w-4 h-4 mr-2" />
              )}
              开始训练
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

export default function TrainingPage() {
  return (
    <WorkspaceLayout currentStep="training">
      <TrainingContent />
    </WorkspaceLayout>
  );
}
