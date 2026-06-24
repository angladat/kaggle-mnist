from pathlib import Path

import torch


class Trainer:

    def __init__(
            self,
            model,
            train_loader,
            val_loader,
            loss_fn,
            optimizer,
        ) -> None:
        """Инициализация"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_one_epoch(self):
        self.model.train()

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.loss_fn(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.loss_fn(logits, labels)

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total
    
    def fit(self, epochs: int, progress: bool = False):
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.evaluate()

            if progress:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"train loss: {train_loss:.4f}, train acc: {train_acc:.4f} | "
                    f"val loss: {val_loss:.4f}, val acc: {val_acc:.4f}"
                )

    def save_checkpoint(self, path: Path):
        torch.save(self.model.state_dict(), path)
