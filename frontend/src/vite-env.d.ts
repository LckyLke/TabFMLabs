/// <reference types="vite/client" />

declare module "*.xlsx?url" {
  const url: string;
  export default url;
}
