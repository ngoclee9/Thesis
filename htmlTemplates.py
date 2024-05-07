css = '''
<style>
.chat-message {
    padding: 1.5rem; 
    display: flex;
    margin-bottom: 1rem;
    border-radius: 0.5rem;
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  object-fit: cover;
  max-width: 78px;
  border-radius: 50%;
  max-height: 78px;
  
  
}
.chat-message .message {
  width: 80%;
  color: #fff;
  padding: 0 1.5rem;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://tse2.mm.bing.net/th?id=OIP.cGqZl9XCvwWU4c0dr-Q_NwHaHa&pid=Api&P=0&h=220" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://tse3.mm.bing.net/th?id=OIP.ve6SGx5xZMXpCVQbtvGyUAHaF8&pid=Api&P=0&h=220">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
