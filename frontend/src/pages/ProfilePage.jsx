import React from 'react';
import '../styles/ProfilePage.css'; // Create this file

const ProfilePage = () => {
  // You can fetch and display user profile information here
  const user = JSON.parse(localStorage.getItem('userData')); // Example

  return (
    <div className="profile-container">
      <h1>Your Profile</h1>
      {user && (
        <div className="profile-info">
          <p><strong>Username:</strong> {user.username}</p>
          <p><strong>Email:</strong> {user.email}</p>
          {/* Add other profile information here */}
        </div>
      )}
      {!user && <p>No user information available.</p>}
      {/* Add more profile details or editing options as needed */}
    </div>
  );
};

export default ProfilePage;