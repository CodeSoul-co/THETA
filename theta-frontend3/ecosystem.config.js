// PM2 配置文件
module.exports = {
  apps: [{
    name: 'theta-frontend',
    script: 'npm',
    args: 'start',
    cwd: '/opt/theta-frontend3',
    instances: 1,
    exec_mode: 'fork',
    env: {
      NODE_ENV: 'production',
      PORT: 3000,
      NEXT_PUBLIC_DATACLEAN_API_URL: 'http://api.yourdomain.com' // 修改为实际 API 地址
    },
    error_file: './logs/err.log',
    out_file: './logs/out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    merge_logs: true,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G'
  }]
}
