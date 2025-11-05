import { Authenticated, Unauthenticated, useQuery } from "convex/react";
import { api } from "../convex/_generated/api";
import { SignInForm } from "./SignInForm";
import { SignOutButton } from "./SignOutButton";
import { Toaster } from "sonner";
import { ImageCaptionGenerator } from "./ImageCaptionGenerator";

export default function App() {
  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-blue-50 to-indigo-100">
      <header className="sticky top-0 z-10 bg-white/90 backdrop-blur-sm h-16 flex justify-between items-center border-b shadow-sm px-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-sm">🖼️</span>
          </div>
          <h2 className="text-xl font-semibold text-gray-800">Image Caption Generator</h2>
        </div>
        <Authenticated>
          <SignOutButton />
        </Authenticated>
      </header>
      <main className="flex-1 p-4">
        <Content />
      </main>
      <Toaster />
    </div>
  );
}

function Content() {
  const loggedInUser = useQuery(api.auth.loggedInUser);

  if (loggedInUser === undefined) {
    return (
      <div className="flex justify-center items-center min-h-[400px]">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto">
      <Authenticated>
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            AI Image Caption Generator
          </h1>
          <p className="text-lg text-gray-600">
            Upload an image and let AI generate a descriptive caption for you
          </p>
        </div>
        <ImageCaptionGenerator />
      </Authenticated>
      
      <Unauthenticated>
        <div className="max-w-md mx-auto mt-20">
          <div className="text-center mb-8">
            <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-white text-2xl">🖼️</span>
            </div>
            <h1 className="text-3xl font-bold text-gray-800 mb-2">
              AI Image Caption Generator
            </h1>
            <p className="text-gray-600">
              Sign in to start generating captions for your images
            </p>
          </div>
          <SignInForm />
        </div>
      </Unauthenticated>
    </div>
  );
}
