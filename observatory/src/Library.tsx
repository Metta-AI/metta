import { useState } from 'react'

// CSS for feed page
const FEED_CSS = `
.feed-container {
  display: flex;
  max-width: 1200px;
  margin: 0 auto;
  min-height: calc(100vh - 60px);
}

.feed-sidebar {
  width: 250px;
  background: #fff;
  border-right: 1px solid #e1e8ed;
  padding: 20px 0;
  position: sticky;
  top: 0;
  height: fit-content;
}

.feed-main {
  flex: 1;
  background: #fff;
}

.feed-header {
  padding: 15px 20px;
  border-bottom: 1px solid #e1e8ed;
  background: #fff;
  position: sticky;
  top: 0;
  z-index: 10;
}

.feed-header h1 {
  font-size: 20px;
  font-weight: 700;
  color: #14171a;
  margin: 0;
}

.nav-item {
  display: flex;
  align-items: center;
  padding: 12px 20px;
  color: #14171a;
  text-decoration: none;
  font-weight: 500;
  font-size: 15px;
  transition: background-color 0.2s ease;
  border-radius: 25px;
  margin: 0 10px 4px 10px;
}

.nav-item:hover {
  background-color: #f7f9fa;
}

.nav-item.active {
  font-weight: 700;
}

.nav-item svg {
  margin-right: 12px;
  width: 20px;
  height: 20px;
}

.post {
  padding: 15px 20px;
  border-bottom: 1px solid #e1e8ed;
  transition: background-color 0.2s ease;
}

.post:hover {
  background-color: #f7f9fa;
}

.post-header {
  display: flex;
  align-items: center;
  margin-bottom: 8px;
}

.post-avatar {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  background: #1da1f2;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: 700;
  font-size: 18px;
  margin-right: 12px;
}

.post-user-info {
  flex: 1;
}

.post-name {
  font-weight: 700;
  color: #14171a;
  font-size: 15px;
  margin: 0;
}

.post-username {
  color: #657786;
  font-size: 14px;
  margin: 0;
}

.post-time {
  color: #657786;
  font-size: 14px;
}

.post-content {
  color: #14171a;
  font-size: 15px;
  line-height: 1.5;
  margin-bottom: 12px;
}

.post-actions {
  display: flex;
  gap: 60px;
  color: #657786;
  font-size: 13px;
}

.post-action {
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  transition: color 0.2s ease;
}

.post-action:hover {
  color: #1da1f2;
}

.post-action svg {
  width: 16px;
  height: 16px;
}

.compose-button {
  background: #1da1f2;
  color: white;
  border: none;
  border-radius: 25px;
  padding: 15px 20px;
  font-weight: 700;
  font-size: 15px;
  cursor: pointer;
  width: calc(100% - 20px);
  margin: 0 10px 20px 10px;
  transition: background-color 0.2s ease;
}

.compose-button:hover {
  background: #1a91da;
}

.trending-section {
  padding: 15px 20px;
  border-bottom: 1px solid #e1e8ed;
}

.trending-title {
  font-weight: 700;
  color: #14171a;
  font-size: 15px;
  margin-bottom: 12px;
}

.trending-item {
  padding: 8px 0;
  cursor: pointer;
}

.trending-topic {
  font-weight: 700;
  color: #14171a;
  font-size: 14px;
  margin-bottom: 2px;
}

.trending-count {
  color: #657786;
  font-size: 12px;
}
`

interface LibraryProps {
  repo: unknown // Using unknown for now since this is a dummy component
}

export function Library({ repo: _repo }: LibraryProps) {
  const [activeNav, setActiveNav] = useState('home')

  // Dummy posts data
  const posts = [
    {
      id: 1,
      name: 'Alice Johnson',
      username: '@alicej',
      avatar: 'A',
      content: 'Just finished implementing a new navigation policy for autonomous vehicles. The results are looking promising! 🚗 #AI #AutonomousVehicles',
      time: '2h',
      likes: 24,
      retweets: 8,
      replies: 3
    },
    {
      id: 2,
      name: 'Bob Chen',
      username: '@bobchen',
      avatar: 'B',
      content: 'Excited to share our latest research on multi-agent coordination policies. The paper is now available on arXiv! 📄 #Research #MultiAgent',
      time: '4h',
      likes: 42,
      retweets: 15,
      replies: 7
    },
    {
      id: 3,
      name: 'Carol Williams',
      username: '@carolw',
      avatar: 'C',
      content: 'Safety protocols are crucial for AI systems. Here\'s a thread on best practices for implementing robust safety policies... 🧵 #AISafety #Policy',
      time: '6h',
      likes: 67,
      retweets: 23,
      replies: 12
    },
    {
      id: 4,
      name: 'David Kim',
      username: '@davidkim',
      avatar: 'D',
      content: 'Working on environmental adaptation policies for different weather conditions. Rain, snow, fog - our models need to handle it all! 🌧️ #WeatherAI',
      time: '8h',
      likes: 31,
      retweets: 9,
      replies: 5
    },
    {
      id: 5,
      name: 'Eva Rodriguez',
      username: '@evarod',
      avatar: 'E',
      content: 'Human-AI interaction policies are fascinating. The balance between automation and human oversight is key to successful deployment. 🤝 #HumanAI',
      time: '10h',
      likes: 53,
      retweets: 18,
      replies: 9
    }
  ]

  const trendingTopics = [
    { topic: '#AISafety', count: '12.5K posts' },
    { topic: '#AutonomousVehicles', count: '8.2K posts' },
    { topic: '#PolicyEvaluation', count: '5.7K posts' },
    { topic: '#MultiAgent', count: '4.1K posts' },
    { topic: '#HumanAI', count: '3.8K posts' }
  ]

  const navItems = [
    { id: 'home', label: 'Home', icon: '🏠' },
    { id: 'explore', label: 'Explore', icon: '🔍' },
    { id: 'notifications', label: 'Notifications', icon: '🔔' },
    { id: 'messages', label: 'Messages', icon: '💬' },
    { id: 'bookmarks', label: 'Bookmarks', icon: '🔖' },
    { id: 'profile', label: 'Profile', icon: '👤' }
  ]

  return (
    <div style={{ fontFamily: 'Arial, sans-serif' }}>
      <style>{FEED_CSS}</style>
      
      <div className="feed-container">
        {/* Left Sidebar */}
        <div className="feed-sidebar">
          <button className="compose-button">
            Post
          </button>
          
          {navItems.map(item => (
            <div
              key={item.id}
              className={`nav-item ${activeNav === item.id ? 'active' : ''}`}
              onClick={() => setActiveNav(item.id)}
            >
              <span>{item.icon}</span>
              {item.label}
            </div>
          ))}
        </div>

        {/* Main Content */}
        <div className="feed-main">
          <div className="feed-header">
            <h1>Home</h1>
          </div>

          {/* Posts Feed */}
          {posts.map(post => (
            <div key={post.id} className="post">
              <div className="post-header">
                <div className="post-avatar">
                  {post.avatar}
                </div>
                <div className="post-user-info">
                  <div className="post-name">{post.name}</div>
                  <div className="post-username">{post.username}</div>
                </div>
                <div className="post-time">{post.time}</div>
              </div>
              
              <div className="post-content">
                {post.content}
              </div>
              
              <div className="post-actions">
                <div className="post-action">
                  <span>💬</span>
                  {post.replies}
                </div>
                <div className="post-action">
                  <span>🔄</span>
                  {post.retweets}
                </div>
                <div className="post-action">
                  <span>❤️</span>
                  {post.likes}
                </div>
                <div className="post-action">
                  <span>📤</span>
                </div>
              </div>
            </div>
          ))}

          {/* Trending Section */}
          <div className="trending-section">
            <div className="trending-title">Trending</div>
            {trendingTopics.map((topic, index) => (
              <div key={index} className="trending-item">
                <div className="trending-topic">{topic.topic}</div>
                <div className="trending-count">{topic.count}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
} 