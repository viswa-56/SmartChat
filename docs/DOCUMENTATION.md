# Coal Power Plant Dashboard Documentation

## 1. Introduction and Problem Statement

The Coal Power Plant Dashboard project aims to provide a comprehensive monitoring and management system for coal power plants. The system addresses the challenges of tracking plant performance, coal usage, emissions, water and ash management, and real-time alerts. It helps plant operators and administrators optimize operations, ensure regulatory compliance, and improve overall efficiency.

## 2. Solution Overview

This project delivers a full-stack web application consisting of a React-based frontend dashboard and a Node.js backend API server. The solution integrates real-time data updates, historical trends, and detailed analytics to provide actionable insights into plant operations. It supports multiple plants and units, offering granular visibility into various operational parameters.

## 3. Features

- Real-time system overview with key metrics such as total plants, generation, efficiency, and capacity utilization.
- Interactive charts displaying generation trends, plant status distribution, load curves, emissions overview, and plant/unit performance.
- Alerts and notifications for active issues.
- Detailed data on coal consumption, quality, ash production, water usage, and emissions.
- User authentication and role-based access control.
- Admin panel for user and data management.
- Data injection service for simulating or importing data.
- Robust error handling and security measures.

## 4. Tech Stack

### Frontend

- **React**: UI library for building interactive user interfaces.
- **Vite**: Fast build tool and development server.
- **Tailwind CSS**: Utility-first CSS framework for styling.
- **Recharts**: Charting library for data visualization.
- **Framer Motion**: Animation library for React.
- **Axios**: HTTP client for API requests.
- **Socket.io-client**: Real-time communication with backend.
- **React Router DOM**: Routing within the React app.
- **React Hook Form & Yup**: Form handling and validation.

### Backend

- **Node.js**: JavaScript runtime environment.
- **Express**: Web framework for building RESTful APIs.
- **PostgreSQL**: Relational database for persistent storage.
- **Sequelize ORM**: Object-relational mapping for database interaction.
- **pg (node-postgres)**: PostgreSQL client for non-auth routes.
- **Socket.io**: Real-time bidirectional event-based communication.
- **Helmet, CORS, Morgan**: Security and logging middleware.
- **Rate Limiting**: To prevent abuse of API endpoints.

## 5. Database Details

### Configuration and Connection

- The database is PostgreSQL, configured with connection pooling for performance.
- Two database connections are used:
  - **pg Pool**: For non-authentication related routes.
  - **Sequelize ORM**: For authentication and related models.

### Key Tables and Models

#### Plant

- Represents a power plant.
- Fields: `id` (UUID), `name`, `location`, `type`, `capacity`, `current_generation`, `status` (online/offline/maintenance), `units` (JSON), `efficiency`, `commission_date`.
- Associations: Has many Units.

#### CoalUsage

- Tracks coal consumption details.
- Fields: `id`, `plant_id` (foreign key), `date`, `amount`, `type` (bituminous, lignite, etc.), `calorific_value`, `ash_content`, `moisture_value`, `sulfur_value`, `supplier`, `cost_per_ton`.
- Includes methods to calculate heat rate, quality index, and total cost.

#### Emission

- Records emission data for plants.
- Fields: `id`, `plant_id` (foreign key), `date`, `type` (SO2, NOx, CO2, PM, etc.), `amount`, `limit`, `compliance` (compliant, warning, non-compliant).
- Hooks to auto-calculate compliance status.
- Validation to ensure emission limits are not critically exceeded.

## 6. API Structure and Services

- RESTful API endpoints organized by domain: auth, alerts, admin, users, data, reports, dashboard, plants, coal, ash, water, emissions.
- API base URL: `/api`
- Supports data retrieval for dashboard metrics, plant details, coal consumption, emissions, ash and water usage.
- Uses Axios on frontend to communicate with backend.
- Real-time updates via Socket.io.

## 7. Real-time Data and Notifications

- Socket.io integration enables real-time communication between frontend and backend.
- Alerts and notifications are pushed to clients when new active alerts occur.
- Real-time emission and plant status updates are supported.

## 8. Error Handling and Security

- Middleware for error handling including Sequelize validation errors, JWT authentication errors, and general server errors.
- Security middleware includes Helmet for HTTP headers, CORS configuration, and rate limiting (configurable).
- Graceful shutdown procedures close database connections properly.

## 9. Data Injection Service

- Admin endpoints to start, stop, and check status of a data injection service.
- Used for simulating or importing data into the system for testing or demonstration purposes.

## 10. Conclusion

This Coal Power Plant Dashboard project provides a robust, scalable, and user-friendly platform for monitoring and managing coal power plant operations. It leverages modern web technologies and best practices to deliver real-time insights, comprehensive analytics, and effective operational control.

---
