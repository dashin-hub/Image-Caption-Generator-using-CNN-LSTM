import { useState, useRef } from "react";
import { useMutation, useQuery } from "convex/react";
import { api } from "../convex/_generated/api";
import { Id } from "../convex/_generated/dataModel";
import { toast } from "sonner";

export function ImageCaptionGenerator() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const generateUploadUrl = useMutation(api.images.generateUploadUrl);
  const saveImage = useMutation(api.images.saveImage);
  const deleteImage = useMutation(api.images.deleteImage);
  const userImages = useQuery(api.images.getUserImages) || [];

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.size > 10 * 1024 * 1024) { // 10MB limit
        toast.error("File size must be less than 10MB");
        return;
      }
      
      if (!file.type.startsWith("image/")) {
        toast.error("Please select an image file");
        return;
      }

      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    try {
      // Get upload URL
      const uploadUrl = await generateUploadUrl();

      // Upload file
      const result = await fetch(uploadUrl, {
        method: "POST",
        headers: { "Content-Type": selectedFile.type },
        body: selectedFile,
      });

      if (!result.ok) {
        throw new Error("Upload failed");
      }

      const { storageId } = await result.json();

      // Save to database
      await saveImage({
        storageId,
        originalName: selectedFile.name,
      });

      // Reset form
      setSelectedFile(null);
      setPreviewUrl(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }

      toast.success("Image uploaded! Generating caption...");
    } catch (error) {
      console.error("Upload error:", error);
      toast.error("Failed to upload image");
    } finally {
      setIsUploading(false);
    }
  };

  const handleDelete = async (imageId: Id<"images">) => {
    try {
      await deleteImage({ imageId });
      toast.success("Image deleted");
    } catch (error) {
      console.error("Delete error:", error);
      toast.error("Failed to delete image");
    }
  };

  return (
    <div className="space-y-8">
      {/* Upload Section */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-semibold text-gray-800 mb-4">Upload New Image</h2>
        
        <div className="space-y-4">
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="hidden"
              id="file-upload"
            />
            <label
              htmlFor="file-upload"
              className="cursor-pointer flex flex-col items-center space-y-2"
            >
              <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
                <span className="text-blue-500 text-xl">📁</span>
              </div>
              <span className="text-gray-600">Click to select an image</span>
              <span className="text-sm text-gray-400">PNG, JPG, GIF up to 10MB</span>
            </label>
          </div>

          {previewUrl && (
            <div className="space-y-4">
              <div className="relative">
                <img
                  src={previewUrl}
                  alt="Preview"
                  className="max-w-full h-64 object-contain mx-auto rounded-lg shadow-md"
                />
              </div>
              <div className="flex gap-2 justify-center">
                <button
                  onClick={handleUpload}
                  disabled={isUploading}
                  className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {isUploading ? "Uploading..." : "Generate Caption"}
                </button>
                <button
                  onClick={() => {
                    setSelectedFile(null);
                    setPreviewUrl(null);
                    if (fileInputRef.current) {
                      fileInputRef.current.value = "";
                    }
                  }}
                  className="px-6 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Images Gallery */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-semibold text-gray-800 mb-4">Your Images</h2>
        
        {userImages.length === 0 ? (
          <div className="text-center py-12">
            <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-gray-400 text-2xl">🖼️</span>
            </div>
            <p className="text-gray-500">No images uploaded yet</p>
            <p className="text-sm text-gray-400">Upload your first image to get started</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {userImages.map((image) => (
              <div key={image._id} className="border rounded-lg overflow-hidden shadow-md hover:shadow-lg transition-shadow">
                <div className="aspect-video bg-gray-100 flex items-center justify-center">
                  {image.url ? (
                    <img
                      src={image.url}
                      alt={image.originalName}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="text-gray-400">Loading...</div>
                  )}
                </div>
                
                <div className="p-4">
                  <h3 className="font-medium text-gray-800 mb-2 truncate">
                    {image.originalName}
                  </h3>
                  
                  <div className="mb-3">
                    {image.isProcessing ? (
                      <div className="flex items-center space-x-2">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                        <span className="text-sm text-gray-500">Generating caption...</span>
                      </div>
                    ) : image.caption ? (
                      <p className="text-sm text-gray-600 leading-relaxed">
                        {image.caption}
                      </p>
                    ) : (
                      <p className="text-sm text-red-500">Caption generation failed</p>
                    )}
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-400">
                      {new Date(image._creationTime).toLocaleDateString()}
                    </span>
                    <button
                      onClick={() => handleDelete(image._id)}
                      className="text-red-500 hover:text-red-700 text-sm transition-colors"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
