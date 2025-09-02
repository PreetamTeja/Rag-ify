// import React, { useState, useEffect } from 'react';
// import { Upload, MessageCircle, User, Trash2, Send, FileText, AlertCircle, CheckCircle, Loader, Plus, Bot, Settings, Sparkles, Zap, BookOpen, Brain } from 'lucide-react';

// // Using proxy, so no need for full URL
// const API_BASE = '';

// export default function App() {
//   const [agents, setAgents] = useState([]);
//   const [selectedAgent, setSelectedAgent] = useState('');
//   const [newAgentName, setNewAgentName] = useState('');
//   const [selectedFile, setSelectedFile] = useState(null);
//   const [uploadLoading, setUploadLoading] = useState(false);
//   const [uploadMessage, setUploadMessage] = useState('');
//   const [uploadError, setUploadError] = useState('');
  
//   const [chatMessages, setChatMessages] = useState([]);
//   const [currentQuestion, setCurrentQuestion] = useState('');
//   const [queryLoading, setQueryLoading] = useState(false);
  
//   const [currentView, setCurrentView] = useState('build'); // 'build' or 'chat'

//   // Load agents on component mount
//   useEffect(() => {
//     loadAgents();
//   }, []);

//   const loadAgents = async () => {
//     try {
//       const response = await fetch(`${API_BASE}/agents`);
//       const data = await response.json();
//       setAgents(data.agents || []);
//     } catch (error) {
//       console.error('Error loading agents:', error);
//     }
//   };

//   const handleFileSelect = (event) => {
//     const file = event.target.files[0];
//     if (file) {
//       const allowedTypes = ['.pdf', '.docx', '.txt'];
//       const fileExtension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
      
//       if (allowedTypes.includes(fileExtension)) {
//         setSelectedFile(file);
//         setUploadError('');
//       } else {
//         setUploadError('Please select a PDF, DOCX, or TXT file');
//         setSelectedFile(null);
//       }
//     }
//   };

//   const handleUpload = async () => {
//     if (!selectedFile || !newAgentName.trim()) {
//       setUploadError('Please select a file and enter an agent name');
//       return;
//     }

//     setUploadLoading(true);
//     setUploadMessage('');
//     setUploadError('');

//     const formData = new FormData();
//     formData.append('file', selectedFile);
//     formData.append('agent_name', newAgentName.trim());

//     try {
//       const response = await fetch(`${API_BASE}/upload`, {
//         method: 'POST',
//         body: formData,
//       });

//       const data = await response.json();

//       if (response.ok) {
//         setUploadMessage(`Successfully uploaded ${data.filename} to agent "${data.agent_name}"`);
//         setSelectedFile(null);
//         setNewAgentName('');
//         await loadAgents();
        
//         // Reset file input
//         document.getElementById('file-input').value = '';
//       } else {
//         setUploadError(data.detail || 'Upload failed');
//       }
//     } catch (error) {
//       setUploadError('Error uploading file: ' + error.message);
//     } finally {
//       setUploadLoading(false);
//     }
//   };

//   const handleQuery = async () => {
//     if (!currentQuestion.trim() || !selectedAgent) {
//       return;
//     }

//     const question = currentQuestion.trim();
//     setCurrentQuestion('');
//     setQueryLoading(true);

//     // Add user message to chat
//     const userMessage = {
//       type: 'user',
//       content: question,
//       timestamp: new Date().toLocaleTimeString()
//     };
//     setChatMessages(prev => [...prev, userMessage]);

//     try {
//       const response = await fetch(`${API_BASE}/query`, {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({
//           question: question,
//           agent_name: selectedAgent,
//         }),
//       });

//       const data = await response.json();

//       // Add AI response to chat
//       const aiMessage = {
//         type: 'ai',
//         content: data.answer,
//         sources: data.sources || [],
//         timestamp: new Date().toLocaleTimeString()
//       };
//       setChatMessages(prev => [...prev, aiMessage]);

//     } catch (error) {
//       const errorMessage = {
//         type: 'ai',
//         content: 'Sorry, I encountered an error while processing your question.',
//         sources: [],
//         timestamp: new Date().toLocaleTimeString()
//       };
//       setChatMessages(prev => [...prev, errorMessage]);
//     } finally {
//       setQueryLoading(false);
//     }
//   };

//   const handleKeyPress = (event) => {
//     if (event.key === 'Enter' && !event.shiftKey) {
//       event.preventDefault();
//       handleQuery();
//     }
//   };

//   const deleteAgent = async (agentName) => {
//     if (!window.confirm(`Are you sure you want to delete agent "${agentName}" and all its documents?`)) {
//       return;
//     }

//     try {
//       const response = await fetch(`${API_BASE}/agents/${agentName}`, {
//         method: 'DELETE',
//       });

//       if (response.ok) {
//         await loadAgents();
//         if (selectedAgent === agentName) {
//           setSelectedAgent('');
//           setChatMessages([]);
//         }
//       }
//     } catch (error) {
//       console.error('Error deleting agent:', error);
//     }
//   };

//   const selectAgentForChat = (agentName) => {
//     setSelectedAgent(agentName);
//     setChatMessages([]);
//     setCurrentView('chat');
//   };

//   const getAgentIcon = (index) => {
//     const icons = [Bot, Brain, Sparkles, Zap, BookOpen, Settings];
//     const IconComponent = icons[index % icons.length];
//     return IconComponent;
//   };

//   // Build Agents View
//   if (currentView === 'build') {
//     return (
//       <div className="min-h-screen bg-gradient-to-br from-orange-50 via-white to-amber-50">
//         {/* Header */}
//         <header className="bg-white/80 backdrop-blur-sm border-b border-orange-100 sticky top-0 z-10">
//           <div className="max-w-7xl mx-auto px-6 py-4">
//             <div className="flex items-center justify-between">
//               <div className="flex items-center gap-3">
//                 <div className="w-8 h-8 bg-gradient-to-br from-orange-400 to-amber-500 rounded-lg flex items-center justify-center">
//                   <Bot className="text-white" size={18} />
//                 </div>
//                 <h1 className="text-xl font-semibold text-gray-900">Agent Builder</h1>
//               </div>
//               <button
//                 onClick={() => setCurrentView('chat')}
//                 className="px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-orange-600 transition-colors flex items-center gap-2"
//               >
//                 <MessageCircle size={16} />
//                 Chat with Agents
//               </button>
//             </div>
//           </div>
//         </header>

//         <div className="max-w-4xl mx-auto px-6 py-12">
//           {/* Hero Section */}
//           <div className="text-center mb-12">
//             <div className="w-16 h-16 bg-gradient-to-br from-orange-400 to-amber-500 rounded-2xl flex items-center justify-center mx-auto mb-6">
//               <Sparkles className="text-white" size={32} />
//             </div>
//             <h1 className="text-4xl font-bold text-gray-900 mb-4">
//               Build Your AI Agents
//             </h1>
//             <p className="text-xl text-gray-600 max-w-2xl mx-auto">
//               Transform your documents into intelligent agents that can answer questions, provide insights, and help you work more efficiently.
//             </p>
//           </div>

//           {/* Upload Card */}
//           <div className="bg-white rounded-2xl shadow-xl border border-orange-100 p-8 mb-8">
//             <div className="flex items-center gap-3 mb-6">
//               <div className="w-10 h-10 bg-gradient-to-br from-orange-400 to-amber-500 rounded-xl flex items-center justify-center">
//                 <Upload className="text-white" size={20} />
//               </div>
//               <h2 className="text-2xl font-semibold text-gray-900">Create New Agent</h2>
//             </div>
            
//             <div className="space-y-6">
//               <div>
//                 <label className="block text-sm font-medium text-gray-700 mb-3">
//                   Agent Name
//                 </label>
//                 <input
//                   type="text"
//                   value={newAgentName}
//                   onChange={(e) => setNewAgentName(e.target.value)}
//                   placeholder="e.g., Legal Assistant, Research Helper, Document Analyzer"
//                   className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent transition-all"
//                 />
//               </div>

//               <div>
//                 <label className="block text-sm font-medium text-gray-700 mb-3">
//                   Document File
//                 </label>
//                 <div className="relative">
//                   <input
//                     id="file-input"
//                     type="file"
//                     accept=".pdf,.docx,.txt"
//                     onChange={handleFileSelect}
//                     className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent transition-all file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-medium file:bg-orange-50 file:text-orange-700 hover:file:bg-orange-100"
//                   />
//                 </div>
//                 <p className="text-sm text-gray-500 mt-2 flex items-center gap-2">
//                   <FileText size={14} />
//                   Supported formats: PDF, DOCX, TXT
//                 </p>
//               </div>

//               {selectedFile && (
//                 <div className="flex items-center gap-3 p-4 bg-green-50 border border-green-200 rounded-xl">
//                   <CheckCircle size={18} className="text-green-600" />
//                   <span className="text-sm text-green-700 font-medium">
//                     Selected: {selectedFile.name}
//                   </span>
//                 </div>
//               )}

//               <button
//                 onClick={handleUpload}
//                 disabled={uploadLoading || !selectedFile || !newAgentName.trim()}
//                 className="w-full bg-gradient-to-r from-orange-500 to-amber-500 text-white py-4 px-6 rounded-xl hover:from-orange-600 hover:to-amber-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3 font-medium transition-all transform hover:scale-[1.02] active:scale-[0.98]"
//               >
//                 {uploadLoading ? (
//                   <>
//                     <Loader className="animate-spin" size={20} />
//                     Processing Document...
//                   </>
//                 ) : (
//                   <>
//                     <Plus size={20} />
//                     Create Agent
//                   </>
//                 )}
//               </button>

//               {uploadMessage && (
//                 <div className="flex items-center gap-3 p-4 bg-green-50 border border-green-200 rounded-xl">
//                   <CheckCircle size={18} className="text-green-600" />
//                   <span className="text-sm text-green-700">{uploadMessage}</span>
//                 </div>
//               )}

//               {uploadError && (
//                 <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-xl">
//                   <AlertCircle size={18} className="text-red-600" />
//                   <span className="text-sm text-red-700">{uploadError}</span>
//                 </div>
//               )}
//             </div>
//           </div>

//           {/* Agents Grid */}
//           <div className="bg-white rounded-2xl shadow-xl border border-orange-100 p-8">
//             <div className="flex items-center gap-3 mb-6">
//               <div className="w-10 h-10 bg-gradient-to-br from-orange-400 to-amber-500 rounded-xl flex items-center justify-center">
//                 <User className="text-white" size={20} />
//               </div>
//               <h2 className="text-2xl font-semibold text-gray-900">Your Agents</h2>
//               <span className="px-3 py-1 bg-orange-100 text-orange-700 rounded-full text-sm font-medium">
//                 {agents.length}
//               </span>
//             </div>
            
//             {agents.length === 0 ? (
//               <div className="text-center py-12">
//                 <Bot size={48} className="mx-auto text-gray-300 mb-4" />
//                 <h3 className="text-xl font-medium text-gray-700 mb-2">
//                   No agents yet
//                 </h3>
//                 <p className="text-gray-500 mb-6">
//                   Create your first AI agent by uploading a document above.
//                 </p>
//               </div>
//             ) : (
//               <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
//                 {agents.map((agent, index) => {
//                   const IconComponent = getAgentIcon(index);
//                   return (
//                     <div
//                       key={agent.name}
//                       className="group relative bg-gradient-to-br from-white to-orange-50 border border-orange-100 rounded-2xl p-6 hover:shadow-lg transition-all duration-300 hover:scale-[1.02]"
//                     >
//                       <div className="flex items-start justify-between mb-4">
//                         <div className="w-12 h-12 bg-gradient-to-br from-orange-400 to-amber-500 rounded-xl flex items-center justify-center">
//                           <IconComponent className="text-white" size={24} />
//                         </div>
//                         <button
//                           onClick={() => deleteAgent(agent.name)}
//                           className="opacity-0 group-hover:opacity-100 p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-all"
//                         >
//                           <Trash2 size={16} />
//                         </button>
//                       </div>
                      
//                       <h3 className="font-semibold text-gray-900 mb-2 text-lg">
//                         {agent.name}
//                       </h3>
//                       <p className="text-gray-600 text-sm mb-4">
//                         {agent.document_count} document chunks available
//                       </p>
                      
//                       <button
//                         onClick={() => selectAgentForChat(agent.name)}
//                         className="w-full bg-gradient-to-r from-orange-500 to-amber-500 text-white py-3 px-4 rounded-xl hover:from-orange-600 hover:to-amber-600 transition-all font-medium flex items-center justify-center gap-2"
//                       >
//                         <MessageCircle size={16} />
//                         Start Chat
//                       </button>
//                     </div>
//                   );
//                 })}
//               </div>
//             )}
//           </div>
//         </div>
//       </div>
//     );
//   }

//   // Chat View
//   return (
//     <div className="min-h-screen bg-gray-50 flex">
//       {/* Sidebar */}
//       <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
//         {/* Sidebar Header */}
//         <div className="p-6 border-b border-gray-200">
//           <div className="flex items-center gap-3 mb-4">
//             <div className="w-8 h-8 bg-gradient-to-br from-orange-400 to-amber-500 rounded-lg flex items-center justify-center">
//               <Bot className="text-white" size={18} />
//             </div>
//             <h1 className="text-lg font-semibold text-gray-900">Agent Chat</h1>
//           </div>
//           <button
//             onClick={() => setCurrentView('build')}
//             className="w-full px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors flex items-center justify-center gap-2 text-sm"
//           >
//             <Plus size={16} />
//             Build New Agent
//           </button>
//         </div>

//         {/* Agent List */}
//         <div className="flex-1 overflow-y-auto p-6">
//           <h3 className="text-sm font-medium text-gray-700 mb-4 uppercase tracking-wide">
//             Select Agent
//           </h3>
//           <div className="space-y-2">
//             {agents.length === 0 ? (
//               <div className="text-center py-8">
//                 <Bot size={32} className="mx-auto text-gray-300 mb-3" />
//                 <p className="text-gray-500 text-sm">No agents available</p>
//                 <p className="text-gray-400 text-xs mt-1">Create one to start chatting</p>
//               </div>
//             ) : (
//               agents.map((agent, index) => {
//                 const IconComponent = getAgentIcon(index);
//                 return (
//                   <button
//                     key={agent.name}
//                     onClick={() => {
//                       setSelectedAgent(agent.name);
//                       setChatMessages([]);
//                     }}
//                     className={`w-full p-4 rounded-xl text-left transition-all hover:bg-orange-50 ${
//                       selectedAgent === agent.name
//                         ? 'bg-gradient-to-r from-orange-100 to-amber-100 border-2 border-orange-200'
//                         : 'border-2 border-transparent hover:border-orange-100'
//                     }`}
//                   >
//                     <div className="flex items-center gap-3">
//                       <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
//                         selectedAgent === agent.name
//                           ? 'bg-gradient-to-br from-orange-400 to-amber-500'
//                           : 'bg-gray-100'
//                       }`}>
//                         <IconComponent 
//                           className={selectedAgent === agent.name ? 'text-white' : 'text-gray-600'} 
//                           size={20} 
//                         />
//                       </div>
//                       <div className="flex-1 min-w-0">
//                         <h4 className="font-medium text-gray-900 truncate">
//                           {agent.name}
//                         </h4>
//                         <p className="text-xs text-gray-500">
//                           {agent.document_count} chunks
//                         </p>
//                       </div>
//                     </div>
//                   </button>
//                 );
//               })
//             )}
//           </div>
//         </div>
//       </div>

//       {/* Main Chat Area */}
//       <div className="flex-1 flex flex-col">
//         {selectedAgent ? (
//           <>
//             {/* Chat Header */}
//             <div className="bg-white border-b border-gray-200 p-6">
//               <div className="flex items-center gap-3">
//                 <div className="w-10 h-10 bg-gradient-to-br from-orange-400 to-amber-500 rounded-xl flex items-center justify-center">
//                   {React.createElement(getAgentIcon(agents.findIndex(a => a.name === selectedAgent)), {
//                     className: "text-white",
//                     size: 20
//                   })}
//                 </div>
//                 <div>
//                   <h2 className="text-xl font-semibold text-gray-900">{selectedAgent}</h2>
//                   <p className="text-sm text-gray-500">AI Agent • Ready to help</p>
//                 </div>
//               </div>
//             </div>

//             {/* Chat Messages */}
//             <div className="flex-1 overflow-y-auto p-6">
//               <div className="max-w-4xl mx-auto space-y-6">
//                 {chatMessages.length === 0 ? (
//                   <div className="text-center py-12">
//                     <div className="w-16 h-16 bg-gradient-to-br from-orange-400 to-amber-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
//                       <MessageCircle className="text-white" size={32} />
//                     </div>
//                     <h3 className="text-xl font-medium text-gray-700 mb-2">
//                       Start a conversation
//                     </h3>
//                     <p className="text-gray-500 max-w-md mx-auto">
//                       Ask {selectedAgent} anything about the uploaded documents. I'm here to help you find information quickly and accurately.
//                     </p>
//                   </div>
//                 ) : (
//                   chatMessages.map((message, index) => (
//                     <div
//                       key={index}
//                       className={`flex ${
//                         message.type === 'user' ? 'justify-end' : 'justify-start'
//                       }`}
//                     >
//                       <div className={`flex gap-3 max-w-3xl ${
//                         message.type === 'user' ? 'flex-row-reverse' : 'flex-row'
//                       }`}>
//                         {/* Avatar */}
//                         <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
//                           message.type === 'user'
//                             ? 'bg-blue-500'
//                             : 'bg-gradient-to-br from-orange-400 to-amber-500'
//                         }`}>
//                           {message.type === 'user' ? (
//                             <User className="text-white" size={16} />
//                           ) : (
//                             React.createElement(getAgentIcon(agents.findIndex(a => a.name === selectedAgent)), {
//                               className: "text-white",
//                               size: 16
//                             })
//                           )}
//                         </div>

//                         {/* Message Content */}
//                         <div className={`flex-1 ${
//                           message.type === 'user' ? 'text-right' : 'text-left'
//                         }`}>
//                           <div className={`inline-block p-4 rounded-2xl ${
//                             message.type === 'user'
//                               ? 'bg-blue-500 text-white'
//                               : 'bg-white border border-gray-200 shadow-sm'
//                           }`}>
//                             <p className="whitespace-pre-wrap leading-relaxed">
//                               {message.content}
//                             </p>
                            
//                             {message.sources && message.sources.length > 0 && (
//                               <div className="mt-3 pt-3 border-t border-gray-200">
//                                 <p className="text-xs text-gray-500 flex items-center gap-2">
//                                   <FileText size={12} />
//                                   Sources: {message.sources.join(', ')}
//                                 </p>
//                               </div>
//                             )}
//                           </div>
//                           <p className={`text-xs text-gray-500 mt-2 ${
//                             message.type === 'user' ? 'text-right' : 'text-left'
//                           }`}>
//                             {message.timestamp}
//                           </p>
//                         </div>
//                       </div>
//                     </div>
//                   ))
//                 )}
                
//                 {queryLoading && (
//                   <div className="flex justify-start">
//                     <div className="flex gap-3 max-w-3xl">
//                       <div className="w-8 h-8 bg-gradient-to-br from-orange-400 to-amber-500 rounded-full flex items-center justify-center flex-shrink-0">
//                         {React.createElement(getAgentIcon(agents.findIndex(a => a.name === selectedAgent)), {
//                           className: "text-white",
//                           size: 16
//                         })}
//                       </div>
//                       <div className="bg-white border border-gray-200 rounded-2xl px-4 py-3 flex items-center gap-3 shadow-sm">
//                         <Loader className="animate-spin text-orange-500" size={16} />
//                         <span className="text-sm text-gray-600">Thinking...</span>
//                       </div>
//                     </div>
//                   </div>
//                 )}
//               </div>
//             </div>

//             {/* Input Area */}
//             <div className="bg-white border-t border-gray-200 p-6">
//               <div className="max-w-4xl mx-auto">
//                 <div className="flex gap-3">
//                   <div className="flex-1 relative">
//                     <input
//                       type="text"
//                       value={currentQuestion}
//                       onChange={(e) => setCurrentQuestion(e.target.value)}
//                       onKeyPress={handleKeyPress}
//                       placeholder={`Ask ${selectedAgent} about the documents...`}
//                       className="w-full px-4 py-4 border border-gray-200 rounded-2xl focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent transition-all pr-12"
//                       disabled={queryLoading}
//                     />
//                   </div>
//                   <button
//                     onClick={handleQuery}
//                     disabled={queryLoading || !currentQuestion.trim()}
//                     className="px-6 py-4 bg-gradient-to-r from-orange-500 to-amber-500 text-white rounded-2xl hover:from-orange-600 hover:to-amber-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition-all transform hover:scale-[1.02] active:scale-[0.98]"
//                   >
//                     {queryLoading ? (
//                       <Loader className="animate-spin" size={18} />
//                     ) : (
//                       <Send size={18} />
//                     )}
//                   </button>
//                 </div>
//               </div>
//             </div>
//           </>
//         ) : (
//           /* No Agent Selected */
//           <div className="flex-1 flex items-center justify-center p-6">
//             <div className="text-center max-w-md">
//               <div className="w-20 h-20 bg-gradient-to-br from-orange-400 to-amber-500 rounded-3xl flex items-center justify-center mx-auto mb-6">
//                 <MessageCircle className="text-white" size={40} />
//               </div>
//               <h3 className="text-2xl font-semibold text-gray-700 mb-3">
//                 Select an Agent
//               </h3>
//               <p className="text-gray-500 mb-6">
//                 Choose an agent from the sidebar to start chatting, or create a new one.
//               </p>
//               <button
//                 onClick={() => setCurrentView('build')}
//                 className="px-6 py-3 bg-gradient-to-r from-orange-500 to-amber-500 text-white rounded-xl hover:from-orange-600 hover:to-amber-600 transition-all font-medium flex items-center gap-2 mx-auto"
//               >
//                 <Plus size={18} />
//                 Create First Agent
//               </button>
//             </div>
//           </div>
//         )}
//       </div>
//     </div>
//   );
// }

import React, { useState, useEffect } from 'react';
import { Upload, MessageCircle, User, Trash2, Send, FileText, AlertCircle, CheckCircle, Loader, Plus, Bot, Settings, Sparkles, Zap, BookOpen, Brain, Edit3, Save, X, Archive, RotateCcw, Clock, MessageSquare } from 'lucide-react';

// Using proxy, so no need for full URL
const API_BASE = '';

export default function App() {
  const [agents, setAgents] = useState([]);
  const [selectedAgent, setSelectedAgent] = useState('');
  const [newAgentName, setNewAgentName] = useState('');
  const [newSystemPrompt, setNewSystemPrompt] = useState('You are a helpful AI assistant that answers questions based on the provided documents. Be accurate, concise, and cite relevant information when possible.');
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState('');
  const [uploadError, setUploadError] = useState('');
  
  const [chatMessages, setChatMessages] = useState([]);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [queryLoading, setQueryLoading] = useState(false);
  
  const [currentView, setCurrentView] = useState('build'); // 'build', 'chat', or 'agent-settings'
  const [chatSessions, setChatSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState('');
  const [isEditingAgent, setIsEditingAgent] = useState(false);
  const [editingSystemPrompt, setEditingSystemPrompt] = useState('');
  const [memoryEnabled, setMemoryEnabled] = useState(true);

  // Load agents on component mount
  useEffect(() => {
    loadAgents();
    loadChatSessions();
  }, []);

  const loadAgents = async () => {
    try {
      const response = await fetch(`${API_BASE}/agents`);
      const data = await response.json();
      setAgents(data.agents || []);
    } catch (error) {
      console.error('Error loading agents:', error);
    }
  };

  const loadChatSessions = () => {
    // In a real app, this would load from backend
    // For now, using localStorage equivalent with in-memory storage
    const sessions = JSON.parse(window.chatSessions || '[]');
    setChatSessions(sessions);
  };

  const saveChatSessions = (sessions) => {
    window.chatSessions = JSON.stringify(sessions);
    setChatSessions(sessions);
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      const allowedTypes = ['.pdf', '.docx', '.txt'];
      const fileExtension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
      
      if (allowedTypes.includes(fileExtension)) {
        setSelectedFile(file);
        setUploadError('');
      } else {
        setUploadError('Please select a PDF, DOCX, or TXT file');
        setSelectedFile(null);
      }
    }
  };

  const handleUpload = async () => {
    if (!selectedFile || !newAgentName.trim()) {
      setUploadError('Please select a file and enter an agent name');
      return;
    }

    setUploadLoading(true);
    setUploadMessage('');
    setUploadError('');

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('agent_name', newAgentName.trim());
    formData.append('system_prompt', newSystemPrompt);

    try {
      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setUploadMessage(`Successfully created agent "${data.agent_name}" with ${data.chunks_processed} document chunks`);
        setSelectedFile(null);
        setNewAgentName('');
        setNewSystemPrompt('You are a helpful AI assistant that answers questions based on the provided documents. Be accurate, concise, and cite relevant information when possible.');
        await loadAgents();
        
        // Reset file input
        document.getElementById('file-input').value = '';
      } else {
        setUploadError(data.detail || 'Upload failed');
      }
    } catch (error) {
      setUploadError('Error uploading file: ' + error.message);
    } finally {
      setUploadLoading(false);
    }
  };

  const updateAgentSystemPrompt = async () => {
    try {
      const response = await fetch(`${API_BASE}/agents/${selectedAgent}/system-prompt`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          system_prompt: editingSystemPrompt
        }),
      });

      if (response.ok) {
        setIsEditingAgent(false);
        // Optionally reload agent data
      }
    } catch (error) {
      console.error('Error updating system prompt:', error);
    }
  };

  const createNewChatSession = () => {
    const sessionId = `session_${Date.now()}`;
    const newSession = {
      id: sessionId,
      agentName: selectedAgent,
      title: `Chat ${chatSessions.filter(s => s.agentName === selectedAgent).length + 1}`,
      messages: [],
      createdAt: new Date().toISOString(),
      lastActivity: new Date().toISOString()
    };
    
    const updatedSessions = [...chatSessions, newSession];
    saveChatSessions(updatedSessions);
    setCurrentSessionId(sessionId);
    setChatMessages([]);
  };

  const selectChatSession = (sessionId) => {
    const session = chatSessions.find(s => s.id === sessionId);
    if (session) {
      setCurrentSessionId(sessionId);
      setChatMessages(session.messages || []);
    }
  };

  const saveCurrentSession = () => {
    if (currentSessionId && chatMessages.length > 0) {
      const updatedSessions = chatSessions.map(session => 
        session.id === currentSessionId 
          ? { 
              ...session, 
              messages: chatMessages,
              lastActivity: new Date().toISOString(),
              title: chatMessages[0]?.content.substring(0, 30) + '...' || session.title
            }
          : session
      );
      saveChatSessions(updatedSessions);
    }
  };

  const deleteSession = (sessionId) => {
    const updatedSessions = chatSessions.filter(s => s.id !== sessionId);
    saveChatSessions(updatedSessions);
    if (currentSessionId === sessionId) {
      setCurrentSessionId('');
      setChatMessages([]);
    }
  };

  const handleQuery = async () => {
    if (!currentQuestion.trim() || !selectedAgent) {
      return;
    }

    const question = currentQuestion.trim();
    setCurrentQuestion('');
    setQueryLoading(true);

    // Add user message to chat
    const userMessage = {
      type: 'user',
      content: question,
      timestamp: new Date().toLocaleTimeString()
    };
    
    const newMessages = [...chatMessages, userMessage];
    setChatMessages(newMessages);

    try {
      const response = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: question,
          agent_name: selectedAgent,
          session_id: currentSessionId,
          memory_enabled: memoryEnabled,
          chat_history: memoryEnabled ? newMessages.slice(-10) : [] // Last 10 messages for context
        }),
      });

      const data = await response.json();

      // Add AI response to chat
      const aiMessage = {
        type: 'ai',
        content: data.answer,
        sources: data.sources || [],
        timestamp: new Date().toLocaleTimeString()
      };
      
      const finalMessages = [...newMessages, aiMessage];
      setChatMessages(finalMessages);
      
      // Save session
      if (currentSessionId) {
        saveCurrentSession();
      }

    } catch (error) {
      const errorMessage = {
        type: 'ai',
        content: 'Sorry, I encountered an error while processing your question.',
        sources: [],
        timestamp: new Date().toLocaleTimeString()
      };
      setChatMessages(prev => [...prev, errorMessage]);
    } finally {
      setQueryLoading(false);
    }
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleQuery();
    }
  };

  const deleteAgent = async (agentName) => {
    if (!window.confirm(`Are you sure you want to delete agent "${agentName}" and all its documents?`)) {
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/agents/${agentName}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        await loadAgents();
        // Clean up sessions for this agent
        const updatedSessions = chatSessions.filter(s => s.agentName !== agentName);
        saveChatSessions(updatedSessions);
        
        if (selectedAgent === agentName) {
          setSelectedAgent('');
          setChatMessages([]);
          setCurrentSessionId('');
        }
      }
    } catch (error) {
      console.error('Error deleting agent:', error);
    }
  };

  const selectAgentForChat = (agentName) => {
    setSelectedAgent(agentName);
    setChatMessages([]);
    setCurrentSessionId('');
    setCurrentView('chat');
  };

  const getAgentIcon = (index) => {
    const icons = [Bot, Brain, Sparkles, Zap, BookOpen, Settings];
    const IconComponent = icons[index % icons.length];
    return IconComponent;
  };

  // Agent Settings View
  if (currentView === 'agent-settings' && selectedAgent) {
    const currentAgent = agents.find(a => a.name === selectedAgent);
    
    return (
      <div className="min-h-screen bg-gray-50">
        <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
          <div className="max-w-7xl mx-auto px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setCurrentView('chat')}
                  className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  <X size={20} className="text-gray-600" />
                </button>
                <div className="w-8 h-8 bg-orange-500 rounded-lg flex items-center justify-center">
                  <Settings className="text-white" size={18} />
                </div>
                <h1 className="text-xl font-semibold text-gray-900">Agent Settings</h1>
              </div>
            </div>
          </div>
        </header>

        <div className="max-w-4xl mx-auto px-6 py-8">
          <div className="bg-white rounded-xl border border-gray-200 p-8">
            <div className="flex items-center gap-3 mb-8">
              <div className="w-12 h-12 bg-orange-500 rounded-xl flex items-center justify-center">
                {React.createElement(getAgentIcon(agents.findIndex(a => a.name === selectedAgent)), {
                  className: "text-white",
                  size: 24
                })}
              </div>
              <div>
                <h2 className="text-2xl font-semibold text-gray-900">{selectedAgent}</h2>
                <p className="text-gray-600">Configure agent behavior and memory</p>
              </div>
            </div>

            <div className="space-y-8">
              {/* System Prompt */}
              <div>
                <div className="flex items-center justify-between mb-4">
                  <label className="block text-sm font-medium text-gray-900">
                    System Prompt
                  </label>
                  {!isEditingAgent ? (
                    <button
                      onClick={() => {
                        setIsEditingAgent(true);
                        setEditingSystemPrompt(currentAgent?.system_prompt || newSystemPrompt);
                      }}
                      className="flex items-center gap-2 px-3 py-1 text-sm text-orange-600 hover:bg-orange-50 rounded-lg transition-colors"
                    >
                      <Edit3 size={16} />
                      Edit
                    </button>
                  ) : (
                    <div className="flex gap-2">
                      <button
                        onClick={updateAgentSystemPrompt}
                        className="flex items-center gap-2 px-3 py-1 text-sm text-green-600 hover:bg-green-50 rounded-lg transition-colors"
                      >
                        <Save size={16} />
                        Save
                      </button>
                      <button
                        onClick={() => setIsEditingAgent(false)}
                        className="flex items-center gap-2 px-3 py-1 text-sm text-gray-600 hover:bg-gray-50 rounded-lg transition-colors"
                      >
                        <X size={16} />
                        Cancel
                      </button>
                    </div>
                  )}
                </div>
                
                {isEditingAgent ? (
                  <textarea
                    value={editingSystemPrompt}
                    onChange={(e) => setEditingSystemPrompt(e.target.value)}
                    className="w-full h-32 px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent resize-none"
                    placeholder="Define how this agent should behave and respond..."
                  />
                ) : (
                  <div className="p-4 bg-gray-50 rounded-xl border border-gray-200">
                    <p className="text-gray-700 text-sm leading-relaxed">
                      {currentAgent?.system_prompt || newSystemPrompt}
                    </p>
                  </div>
                )}
              </div>

              {/* Memory Settings */}
              <div>
                <label className="block text-sm font-medium text-gray-900 mb-4">
                  Memory & Context
                </label>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 bg-gray-50 rounded-xl border border-gray-200">
                    <div>
                      <h4 className="font-medium text-gray-900">Chat Memory</h4>
                      <p className="text-sm text-gray-600">Remember conversation context between messages</p>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        checked={memoryEnabled}
                        onChange={(e) => setMemoryEnabled(e.target.checked)}
                        className="sr-only peer"
                      />
                      <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-orange-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-orange-500"></div>
                    </label>
                  </div>
                </div>
              </div>

              {/* Agent Stats */}
              <div>
                <label className="block text-sm font-medium text-gray-900 mb-4">
                  Agent Statistics
                </label>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="p-4 bg-gray-50 rounded-xl border border-gray-200 text-center">
                    <FileText className="w-8 h-8 text-orange-500 mx-auto mb-2" />
                    <p className="text-2xl font-semibold text-gray-900">{currentAgent?.document_count || 0}</p>
                    <p className="text-sm text-gray-600">Document Chunks</p>
                  </div>
                  <div className="p-4 bg-gray-50 rounded-xl border border-gray-200 text-center">
                    <MessageSquare className="w-8 h-8 text-orange-500 mx-auto mb-2" />
                    <p className="text-2xl font-semibold text-gray-900">{chatSessions.filter(s => s.agentName === selectedAgent).length}</p>
                    <p className="text-sm text-gray-600">Chat Sessions</p>
                  </div>
                  <div className="p-4 bg-gray-50 rounded-xl border border-gray-200 text-center">
                    <Clock className="w-8 h-8 text-orange-500 mx-auto mb-2" />
                    <p className="text-2xl font-semibold text-gray-900">
                      {currentAgent ? new Date(currentAgent.created_at || Date.now()).toLocaleDateString() : '-'}
                    </p>
                    <p className="text-sm text-gray-600">Created</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Build Agents View
  if (currentView === 'build') {
    return (
      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
          <div className="max-w-7xl mx-auto px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-orange-500 rounded-lg flex items-center justify-center">
                  <Bot className="text-white" size={18} />
                </div>
                <h1 className="text-xl font-semibold text-gray-900">Agent Builder</h1>
              </div>
              <button
                onClick={() => setCurrentView('chat')}
                className="px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-orange-600 transition-colors flex items-center gap-2"
              >
                <MessageCircle size={16} />
                Chat with Agents
              </button>
            </div>
          </div>
        </header>

        <div className="max-w-4xl mx-auto px-6 py-12">
          {/* Hero Section */}
          <div className="text-center mb-12">
            <div className="w-16 h-16 bg-orange-500 rounded-2xl flex items-center justify-center mx-auto mb-6">
              <Sparkles className="text-white" size={32} />
            </div>
            <h1 className="text-4xl font-bold text-gray-900 mb-4">
              Build Your AI Agents
            </h1>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Transform your documents into intelligent agents that can answer questions, provide insights, and help you work more efficiently.
            </p>
          </div>

          {/* Upload Card */}
          <div className="bg-white rounded-xl border border-gray-200 p-8 mb-8 shadow-sm">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 bg-orange-500 rounded-xl flex items-center justify-center">
                <Upload className="text-white" size={20} />
              </div>
              <h2 className="text-2xl font-semibold text-gray-900">Create New Agent</h2>
            </div>
            
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-3">
                  Agent Name
                </label>
                <input
                  type="text"
                  value={newAgentName}
                  onChange={(e) => setNewAgentName(e.target.value)}
                  placeholder="e.g., Legal Assistant, Research Helper, Document Analyzer"
                  className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent transition-all"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-3">
                  System Prompt
                </label>
                <textarea
                  value={newSystemPrompt}
                  onChange={(e) => setNewSystemPrompt(e.target.value)}
                  placeholder="Define how this agent should behave and respond to questions..."
                  className="w-full h-24 px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent resize-none transition-all"
                />
                <p className="text-sm text-gray-500 mt-2">
                  This defines the personality and behavior of your agent
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-3">
                  Document File
                </label>
                <div className="relative">
                  <input
                    id="file-input"
                    type="file"
                    accept=".pdf,.docx,.txt"
                    onChange={handleFileSelect}
                    className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent transition-all file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-medium file:bg-orange-50 file:text-orange-700 hover:file:bg-orange-100"
                  />
                </div>
                <p className="text-sm text-gray-500 mt-2 flex items-center gap-2">
                  <FileText size={14} />
                  Supported formats: PDF, DOCX, TXT
                </p>
              </div>

              {selectedFile && (
                <div className="flex items-center gap-3 p-4 bg-green-50 border border-green-200 rounded-xl">
                  <CheckCircle size={18} className="text-green-600" />
                  <span className="text-sm text-green-700 font-medium">
                    Selected: {selectedFile.name}
                  </span>
                </div>
              )}

              <button
                onClick={handleUpload}
                disabled={uploadLoading || !selectedFile || !newAgentName.trim()}
                className="w-full bg-orange-500 text-white py-4 px-6 rounded-xl hover:bg-orange-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3 font-medium transition-all"
              >
                {uploadLoading ? (
                  <>
                    <Loader className="animate-spin" size={20} />
                    Processing Document...
                  </>
                ) : (
                  <>
                    <Plus size={20} />
                    Create Agent
                  </>
                )}
              </button>

              {uploadMessage && (
                <div className="flex items-center gap-3 p-4 bg-green-50 border border-green-200 rounded-xl">
                  <CheckCircle size={18} className="text-green-600" />
                  <span className="text-sm text-green-700">{uploadMessage}</span>
                </div>
              )}

              {uploadError && (
                <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-xl">
                  <AlertCircle size={18} className="text-red-600" />
                  <span className="text-sm text-red-700">{uploadError}</span>
                </div>
              )}
            </div>
          </div>

          {/* Agents Grid */}
          <div className="bg-white rounded-xl border border-gray-200 p-8 shadow-sm">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 bg-orange-500 rounded-xl flex items-center justify-center">
                <User className="text-white" size={20} />
              </div>
              <h2 className="text-2xl font-semibold text-gray-900">Your Agents</h2>
              <span className="px-3 py-1 bg-orange-100 text-orange-700 rounded-full text-sm font-medium">
                {agents.length}
              </span>
            </div>
            
            {agents.length === 0 ? (
              <div className="text-center py-12">
                <Bot size={48} className="mx-auto text-gray-300 mb-4" />
                <h3 className="text-xl font-medium text-gray-700 mb-2">
                  No agents yet
                </h3>
                <p className="text-gray-500 mb-6">
                  Create your first AI agent by uploading a document above.
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {agents.map((agent, index) => {
                  const IconComponent = getAgentIcon(index);
                  return (
                    <div
                      key={agent.name}
                      className="group relative bg-white border border-gray-200 rounded-xl p-6 hover:shadow-md hover:border-orange-200 transition-all duration-300"
                    >
                      <div className="flex items-start justify-between mb-4">
                        <div className="w-12 h-12 bg-orange-500 rounded-xl flex items-center justify-center">
                          <IconComponent className="text-white" size={24} />
                        </div>
                        <button
                          onClick={() => deleteAgent(agent.name)}
                          className="opacity-0 group-hover:opacity-100 p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-all"
                        >
                          <Trash2 size={16} />
                        </button>
                      </div>
                      
                      <h3 className="font-semibold text-gray-900 mb-2 text-lg">
                        {agent.name}
                      </h3>
                      <p className="text-gray-600 text-sm mb-4">
                        {agent.document_count} document chunks available
                      </p>
                      
                      <div className="space-y-2">
                        <button
                          onClick={() => selectAgentForChat(agent.name)}
                          className="w-full bg-orange-500 text-white py-3 px-4 rounded-xl hover:bg-orange-600 transition-all font-medium flex items-center justify-center gap-2"
                        >
                          <MessageCircle size={16} />
                          Start Chat
                        </button>
                        <button
                          onClick={() => {
                            setSelectedAgent(agent.name);
                            setCurrentView('agent-settings');
                          }}
                          className="w-full bg-gray-100 text-gray-700 py-2 px-4 rounded-lg hover:bg-gray-200 transition-all text-sm flex items-center justify-center gap-2"
                        >
                          <Settings size={14} />
                          Settings
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Chat View
  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* Sidebar */}
      <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
        {/* Sidebar Header */}
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-8 h-8 bg-orange-500 rounded-lg flex items-center justify-center">
              <Bot className="text-white" size={18} />
            </div>
            <h1 className="text-lg font-semibold text-gray-900">Agent Chat</h1>
          </div>
          <button
            onClick={() => setCurrentView('build')}
            className="w-full px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors flex items-center justify-center gap-2 text-sm mb-3"
          >
            <Plus size={16} />
            Build New Agent
          </button>
          
          {selectedAgent && (
            <button
              onClick={createNewChatSession}
              className="w-full px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-orange-600 transition-colors flex items-center justify-center gap-2 text-sm"
            >
              <MessageCircle size={16} />
              New Chat
            </button>
          )}
        </div>

        {/* Agent List */}
        <div className="flex-1 overflow-y-auto">
          {/* Current Agent */}
          {selectedAgent && (
            <div className="p-4 border-b border-gray-100">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium text-gray-700 uppercase tracking-wide">
                  Current Agent
                </h3>
                <button
                  onClick={() => {
                    setCurrentView('agent-settings');
                  }}
                  className="p-1 hover:bg-gray-100 rounded transition-colors"
                >
                  <Settings size={16} className="text-gray-500" />
                </button>
              </div>
              
              <div className="p-3 bg-orange-50 border border-orange-200 rounded-lg">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-orange-500 rounded-lg flex items-center justify-center">
                    {React.createElement(getAgentIcon(agents.findIndex(a => a.name === selectedAgent)), {
                      className: "text-white",
                      size: 16
                    })}
                  </div>
                  <div className="flex-1">
                    <h4 className="font-medium text-gray-900 text-sm">{selectedAgent}</h4>
                    <p className="text-xs text-gray-600">
                      {agents.find(a => a.name === selectedAgent)?.document_count || 0} chunks
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Chat Sessions */}
          <div className="p-4">
            <h3 className="text-sm font-medium text-gray-700 uppercase tracking-wide mb-3">
              Chat History
            </h3>
            
            {selectedAgent ? (
              <div className="space-y-2">
                {chatSessions
                  .filter(session => session.agentName === selectedAgent)
                  .sort((a, b) => new Date(b.lastActivity) - new Date(a.lastActivity))
                  .map((session) => (
                    <div
                      key={session.id}
                      className={`group p-3 rounded-lg cursor-pointer transition-all ${
                        currentSessionId === session.id
                          ? 'bg-orange-100 border-orange-200'
                          : 'hover:bg-gray-50 border-transparent'
                      } border`}
                      onClick={() => selectChatSession(session.id)}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1 min-w-0">
                          <h4 className="text-sm font-medium text-gray-900 truncate">
                            {session.title}
                          </h4>
                          <p className="text-xs text-gray-500 mt-1">
                            {new Date(session.lastActivity).toLocaleDateString()}
                          </p>
                          <p className="text-xs text-gray-400">
                            {session.messages?.length || 0} messages
                          </p>
                        </div>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteSession(session.id);
                          }}
                          className="opacity-0 group-hover:opacity-100 p-1 text-gray-400 hover:text-red-500 transition-all"
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </div>
                  ))}
                
                {chatSessions.filter(s => s.agentName === selectedAgent).length === 0 && (
                  <div className="text-center py-6">
                    <Archive size={24} className="mx-auto text-gray-300 mb-2" />
                    <p className="text-sm text-gray-500">No chat history</p>
                    <p className="text-xs text-gray-400">Start a new conversation</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-6">
                <MessageCircle size={24} className="mx-auto text-gray-300 mb-2" />
                <p className="text-sm text-gray-500">Select an agent first</p>
              </div>
            )}
          </div>

          {/* Available Agents */}
          <div className="p-4 border-t border-gray-100">
            <h3 className="text-sm font-medium text-gray-700 uppercase tracking-wide mb-3">
              All Agents
            </h3>
            <div className="space-y-2">
              {agents.length === 0 ? (
                <div className="text-center py-6">
                  <Bot size={24} className="mx-auto text-gray-300 mb-2" />
                  <p className="text-sm text-gray-500">No agents available</p>
                  <p className="text-xs text-gray-400">Create one to start chatting</p>
                </div>
              ) : (
                agents.map((agent, index) => {
                  const IconComponent = getAgentIcon(index);
                  return (
                    <button
                      key={agent.name}
                      onClick={() => {
                        setSelectedAgent(agent.name);
                        setChatMessages([]);
                        setCurrentSessionId('');
                      }}
                      className={`w-full p-3 rounded-lg text-left transition-all hover:bg-gray-50 ${
                        selectedAgent === agent.name
                          ? 'bg-orange-50 border border-orange-200'
                          : 'border border-transparent'
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                          selectedAgent === agent.name ? 'bg-orange-500' : 'bg-gray-100'
                        }`}>
                          <IconComponent 
                            className={selectedAgent === agent.name ? 'text-white' : 'text-gray-600'} 
                            size={16} 
                          />
                        </div>
                        <div className="flex-1 min-w-0">
                          <h4 className="font-medium text-gray-900 truncate text-sm">
                            {agent.name}
                          </h4>
                          <p className="text-xs text-gray-500">
                            {agent.document_count} chunks
                          </p>
                        </div>
                      </div>
                    </button>
                  );
                })
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {selectedAgent ? (
          <>
            {/* Chat Header */}
            <div className="bg-white border-b border-gray-200 p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-orange-500 rounded-xl flex items-center justify-center">
                    {React.createElement(getAgentIcon(agents.findIndex(a => a.name === selectedAgent)), {
                      className: "text-white",
                      size: 20
                    })}
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-gray-900">{selectedAgent}</h2>
                    <div className="flex items-center gap-2">
                      <p className="text-sm text-gray-500">AI Agent</p>
                      {memoryEnabled && (
                        <span className="px-2 py-1 bg-green-100 text-green-700 rounded text-xs font-medium">
                          Memory On
                        </span>
                      )}
                      {currentSessionId && (
                        <span className="px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs font-medium">
                          Session Active
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setCurrentView('agent-settings')}
                    className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                  >
                    <Settings size={18} />
                  </button>
                </div>
              </div>
            </div>

            {/* Chat Messages */}
            <div className="flex-1 overflow-y-auto p-6 bg-gray-50">
              <div className="max-w-4xl mx-auto space-y-6">
                {chatMessages.length === 0 ? (
                  <div className="text-center py-12">
                    <div className="w-16 h-16 bg-orange-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
                      <MessageCircle className="text-white" size={32} />
                    </div>
                    <h3 className="text-xl font-medium text-gray-700 mb-2">
                      {currentSessionId ? 'Continue your conversation' : 'Start a new conversation'}
                    </h3>
                    <p className="text-gray-500 max-w-md mx-auto mb-6">
                      Ask {selectedAgent} anything about the uploaded documents. I'm here to help you find information quickly and accurately.
                    </p>
                    {!currentSessionId && (
                      <button
                        onClick={createNewChatSession}
                        className="px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-orange-600 transition-colors flex items-center gap-2 mx-auto"
                      >
                        <Plus size={16} />
                        Start New Session
                      </button>
                    )}
                  </div>
                ) : (
                  chatMessages.map((message, index) => (
                    <div
                      key={index}
                      className={`flex ${
                        message.type === 'user' ? 'justify-end' : 'justify-start'
                      }`}
                    >
                      <div className={`flex gap-3 max-w-3xl ${
                        message.type === 'user' ? 'flex-row-reverse' : 'flex-row'
                      }`}>
                        {/* Avatar */}
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                          message.type === 'user'
                            ? 'bg-blue-500'
                            : 'bg-orange-500'
                        }`}>
                          {message.type === 'user' ? (
                            <User className="text-white" size={16} />
                          ) : (
                            React.createElement(getAgentIcon(agents.findIndex(a => a.name === selectedAgent)), {
                              className: "text-white",
                              size: 16
                            })
                          )}
                        </div>

                        {/* Message Content */}
                        <div className={`flex-1 ${
                          message.type === 'user' ? 'text-right' : 'text-left'
                        }`}>
                          <div className={`inline-block p-4 rounded-2xl max-w-full ${
                            message.type === 'user'
                              ? 'bg-blue-500 text-white'
                              : 'bg-white border border-gray-200 shadow-sm'
                          }`}>
                            <p className="whitespace-pre-wrap leading-relaxed text-sm">
                              {message.content}
                            </p>
                            
                            {message.sources && message.sources.length > 0 && (
                              <div className="mt-3 pt-3 border-t border-gray-200">
                                <p className="text-xs text-gray-500 flex items-center gap-2">
                                  <FileText size={12} />
                                  Sources: {message.sources.join(', ')}
                                </p>
                              </div>
                            )}
                          </div>
                          <p className={`text-xs text-gray-500 mt-2 ${
                            message.type === 'user' ? 'text-right' : 'text-left'
                          }`}>
                            {message.timestamp}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))
                )}
                
                {queryLoading && (
                  <div className="flex justify-start">
                    <div className="flex gap-3 max-w-3xl">
                      <div className="w-8 h-8 bg-orange-500 rounded-full flex items-center justify-center flex-shrink-0">
                        {React.createElement(getAgentIcon(agents.findIndex(a => a.name === selectedAgent)), {
                          className: "text-white",
                          size: 16
                        })}
                      </div>
                      <div className="bg-white border border-gray-200 rounded-2xl px-4 py-3 flex items-center gap-3 shadow-sm">
                        <Loader className="animate-spin text-orange-500" size={16} />
                        <span className="text-sm text-gray-600">Thinking...</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Input Area */}
            <div className="bg-white border-t border-gray-200 p-4">
              <div className="max-w-4xl mx-auto">
                {!currentSessionId && chatMessages.length === 0 && (
                  <div className="mb-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
                    <p className="text-sm text-amber-700 flex items-center gap-2">
                      <AlertCircle size={16} />
                      Start a new chat session to begin conversation with {selectedAgent}
                    </p>
                  </div>
                )}
                
                <div className="flex gap-3">
                  <div className="flex-1 relative">
                    <input
                      type="text"
                      value={currentQuestion}
                      onChange={(e) => setCurrentQuestion(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder={currentSessionId ? `Ask ${selectedAgent} about the documents...` : 'Start a new session to chat...'}
                      className="w-full px-4 py-4 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent transition-all"
                      disabled={queryLoading || !currentSessionId}
                    />
                  </div>
                  <button
                    onClick={currentSessionId ? handleQuery : createNewChatSession}
                    disabled={queryLoading || (!currentQuestion.trim() && currentSessionId)}
                    className="px-6 py-4 bg-orange-500 text-white rounded-xl hover:bg-orange-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition-all"
                  >
                    {queryLoading ? (
                      <Loader className="animate-spin" size={18} />
                    ) : currentSessionId ? (
                      <Send size={18} />
                    ) : (
                      <Plus size={18} />
                    )}
                  </button>
                </div>
              </div>
            </div>
          </>
        ) : (
          /* No Agent Selected */
          <div className="flex-1 flex items-center justify-center p-6">
            <div className="text-center max-w-md">
              <div className="w-20 h-20 bg-orange-500 rounded-3xl flex items-center justify-center mx-auto mb-6">
                <MessageCircle className="text-white" size={40} />
              </div>
              <h3 className="text-2xl font-semibold text-gray-700 mb-3">
                Select an Agent
              </h3>
              <p className="text-gray-500 mb-6">
                Choose an agent from the sidebar to start chatting, or create a new one.
              </p>
              <button
                onClick={() => setCurrentView('build')}
                className="px-6 py-3 bg-orange-500 text-white rounded-xl hover:bg-orange-600 transition-all font-medium flex items-center gap-2 mx-auto"
              >
                <Plus size={18} />
                Create First Agent
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}