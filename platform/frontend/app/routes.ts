import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [
  index("routes/landing.tsx"),
  route("app", "routes/home.tsx"),
  route("app/:conversationId", "routes/chat.tsx"),
  route("login", "routes/login.tsx"),
  route("register", "routes/register.tsx"),
  route("scenarios", "routes/scenarios.tsx"),
  route("scenarios/:scenarioId", "routes/scenario-editor.tsx"),
  route("connections", "routes/connections.tsx"),
  route("history", "routes/history.tsx"),
  route("runs/:jobId", "routes/run-detail.tsx"),
  route("settings", "routes/settings.tsx"),
  route("api/*", "routes/api.ts"),
] satisfies RouteConfig;
