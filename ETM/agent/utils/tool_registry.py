"""
工具注册表 (Tool Registry)

管理Agent可用的工具，包括注册、调用和描述。
"""

from typing import Dict, Any, List, Callable, Optional


class ToolRegistry:
    """
    工具注册表，管理Agent可用的工具。
    """
    
    def __init__(self):
        """
        初始化工具注册表。
        """
        self.tools = {}
        self.descriptions = {}
    
    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str
    ) -> None:
        """
        注册工具。
        
        Args:
            name: 工具名称
            func: 工具函数
            description: 工具描述
        """
        self.tools[name] = func
        self.descriptions[name] = description
    
    def call_tool(
        self,
        name: str,
        args: Dict[str, Any]
    ) -> Any:
        """
        调用工具。
        
        Args:
            name: 工具名称
            args: 工具参数
            
        Returns:
            工具调用结果
        """
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found")
        
        return self.tools[name](**args)
    
    def get_tool_description(
        self,
        name: str
    ) -> str:
        """
        获取工具描述。
        
        Args:
            name: 工具名称
            
        Returns:
            工具描述
        """
        if name not in self.descriptions:
            raise ValueError(f"Tool '{name}' not found")
        
        return self.descriptions[name]
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """
        获取所有工具。
        
        Returns:
            工具列表
        """
        return [
            {
                "name": name,
                "description": self.descriptions[name]
            }
            for name in self.tools
        ]
    
    def register_default_tools(self) -> None:
        """
        注册默认工具。
        """
        # 搜索工具
        self.register_tool(
            name="search",
            func=self._search_tool,
            description="搜索相关信息"
        )
        
        # 计算工具
        self.register_tool(
            name="calculate",
            func=self._calculate_tool,
            description="执行数学计算"
        )
        
        # 日期工具
        self.register_tool(
            name="get_date",
            func=self._date_tool,
            description="获取当前日期和时间"
        )
    
    def _search_tool(self, query: str) -> Dict[str, Any]:
        """
        搜索工具实现。
        
        Args:
            query: 搜索查询
            
        Returns:
            搜索结果
        """
        # 这里是一个简单的模拟实现
        # 实际应用中，可以接入搜索API
        return {
            "query": query,
            "results": [
                {"title": f"Result for {query} 1", "snippet": f"This is a snippet about {query}..."},
                {"title": f"Result for {query} 2", "snippet": f"More information about {query}..."}
            ]
        }
    
    def _calculate_tool(self, expression: str) -> Dict[str, Any]:
        """
        计算工具实现。
        
        Args:
            expression: 数学表达式
            
        Returns:
            计算结果
        """
        try:
            # 安全的eval实现
            import ast
            import operator
            
            # 支持的操作符
            operators = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.BitXor: operator.xor,
                ast.USub: operator.neg
            }
            
            def eval_expr(expr):
                return eval_(ast.parse(expr, mode='eval').body)
            
            def eval_(node):
                if isinstance(node, ast.Num):
                    return node.n
                elif isinstance(node, ast.BinOp):
                    return operators[type(node.op)](eval_(node.left), eval_(node.right))
                elif isinstance(node, ast.UnaryOp):
                    return operators[type(node.op)](eval_(node.operand))
                else:
                    raise TypeError(node)
            
            result = eval_expr(expression)
            
            return {
                "expression": expression,
                "result": result,
                "success": True
            }
        except Exception as e:
            return {
                "expression": expression,
                "error": str(e),
                "success": False
            }
    
    def _date_tool(self) -> Dict[str, Any]:
        """
        日期工具实现。
        
        Returns:
            当前日期和时间
        """
        import datetime
        
        now = datetime.datetime.now()
        
        return {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "day_of_week": now.strftime("%A"),
            "timestamp": now.timestamp()
        }
